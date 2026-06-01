%% ISP YUV420 自适应锐化单文件脚本
% 链路：
% RGB -> YUV420 -> Y域内容分析 -> 频率分解 -> 自适应锐化
% -> halo/过冲控制 -> 感知重建 -> YUV420转回RGB
%
% 不使用 OpenCV / imfilter / fspecial。
% 卷积核、高斯核、Sobel、Laplacian、局部统计均在本脚本中自定义。

clear; clc; close all;

%% ========== 1. 路径配置 ==========
inputPath  = 'D:\project\isp-python-8bit-rgb\input\17.png';
outputPath = 'D:\project\isp-python-8bit-rgb\output\17_sharpen_matlab.png';
debugDir   = 'D:\project\isp-python-8bit-rgb\debug_matlab';

%% ========== 2. 锐化参数 ==========
cfg.strength = 1.0;          % 总锐化强度
cfg.textureBoost = 0.85;     % 纹理增强强度
cfg.edgeBoost = 0.65;        % 结构边缘增强强度
cfg.fineBoost = 0.35;        % 高频细节参与比例
cfg.noiseRejection = 0.85;   % 噪声抑制强度
cfg.haloGuard = 0.75;        % halo 抑制强度
cfg.overshootMargin = 3.0;   % 局部过冲容忍范围

%% ========== 3. 读取 RGB 并转换到 YUV420 ==========
rgb = imread(inputPath);

if size(rgb, 3) ~= 3
    error('输入图像必须是 RGB 三通道图像。');
end

rgb = uint8(rgb);
frame = rgb_to_yuv420(rgb);

Y = frame.y;

%% ========== 4. 内容分析 ==========
edgeInfo = edge_analyze(Y);
textureInfo = texture_analyze(Y);
structureInfo = structure_analyze(edgeInfo.gx, edgeInfo.gy, edgeInfo.edge);
noiseInfo = noise_analyze(Y, edgeInfo.edge, textureInfo.texture);

%% ========== 5. 频率分解 ==========
freq = frequency_decompose(Y);

%% ========== 6. 自适应增强 ==========
enhanceInfo = adaptive_enhance(freq, edgeInfo, textureInfo, structureInfo, noiseInfo, cfg);

%% ========== 7. halo 与过冲控制 ==========
artifactInfo = artifact_control(Y, enhanceInfo.candidateY, edgeInfo, textureInfo, cfg);

%% ========== 8. 感知重建 ==========
Yout = perceptual_reconstruct(Y, artifactInfo.limitedY, textureInfo, structureInfo, noiseInfo);
Yout = min(max(Yout, 0), 255);

%% ========== 9. YUV420 转回 RGB ==========
frame.y = Yout;
outRgb = yuv420_to_rgb(frame);

ensure_parent_dir(outputPath);
imwrite(outRgb, outputPath);

%% ========== 10. 保存中间图 ==========
if ~exist(debugDir, 'dir')
    mkdir(debugDir);
end

save_debug(debugDir, '00_input_rgb.png', rgb);
save_debug(debugDir, '01_y_input.png', Y);
save_debug(debugDir, '01_u_420.png', frame.u);
save_debug(debugDir, '01_v_420.png', frame.v);
save_debug(debugDir, '02_edge_map.png', edgeInfo.edge);
save_debug(debugDir, '03_texture_map.png', textureInfo.texture);
save_debug(debugDir, '04_structure_map.png', structureInfo.structure);
save_debug(debugDir, '05_noise_likelihood.png', noiseInfo.noiseLikelihood);
save_debug(debugDir, '06_base_luma.png', freq.base);
save_debug(debugDir, '07_mid_detail.png', freq.mid);
save_debug(debugDir, '08_fine_detail.png', freq.fine);
save_debug(debugDir, '09_adaptive_detail.png', enhanceInfo.adaptiveDetail);
save_debug(debugDir, '10_candidate_y.png', enhanceInfo.candidateY);
save_debug(debugDir, '11_halo_risk.png', artifactInfo.haloRisk);
save_debug(debugDir, '12_limited_y.png', artifactInfo.limitedY);
save_debug(debugDir, '13_y_output.png', Yout);
save_debug(debugDir, '14_y_delta.png', Yout - Y);
save_debug(debugDir, '15_output_rgb.png', outRgb);

fprintf('锐化结果已保存：%s\n', outputPath);
fprintf('中间图已保存到：%s\n', debugDir);

%% ========================================================================
%% 子函数 1：边缘分析
%% ========================================================================
function edgeInfo = edge_analyze(Y)
    % 使用自定义 Sobel 核计算 Y 通道边缘强度。
    % 公式：
    % Gx = Y * SobelX
    % Gy = Y * SobelY
    % Edge = sqrt(Gx^2 + Gy^2)
    % 最后使用 98 分位数归一化到 0~1。

    SobelX = [-1 0 1;
              -2 0 2;
              -1 0 1] / 8.0;

    SobelY = SobelX';

    gx = conv2_custom(Y, SobelX);
    gy = conv2_custom(Y, SobelY);

    grad = sqrt(gx.^2 + gy.^2);
    edge = normalize01(grad, 98.0);

    edgeInfo.gx = gx;
    edgeInfo.gy = gy;
    edgeInfo.grad = grad;
    edgeInfo.edge = edge;
end

%% ========================================================================
%% 子函数 2：纹理分析
%% ========================================================================
function textureInfo = texture_analyze(Y)
    % 使用局部方差估计纹理强度。
    % 局部方差：
    % Var = E(Y^2) - E(Y)^2
    %
    % 平坦区域：Var 小
    % 纹理区域：Var 大
    % 噪声区域：Var 也可能大，所以后面还要结合噪声似然抑制。

    localMean = box_mean(Y, 7);
    localMeanSq = box_mean(Y.^2, 7);

    localVar = max(localMeanSq - localMean.^2, 0);
    textureEnergy = sqrt(localVar);
    texture = normalize01(textureEnergy, 98.0);

    textureInfo.localMean = localMean;
    textureInfo.localVar = localVar;
    textureInfo.textureEnergy = textureEnergy;
    textureInfo.texture = texture;
end

%% ========================================================================
%% 子函数 3：结构分析
%% ========================================================================
function structureInfo = structure_analyze(gx, gy, edge)
    % 使用结构张量判断边缘方向是否一致。
    %
    % J = [Gx^2  GxGy
    %      GxGy  Gy^2]
    %
    % coherence 越高，说明局部梯度方向越一致，更像真实结构边缘。
    % 随机噪声的方向通常更混乱，coherence 不会稳定偏高。

    jxx = blur_gaussian(gx.^2, 5, 1.0);
    jyy = blur_gaussian(gy.^2, 5, 1.0);
    jxy = blur_gaussian(gx .* gy, 5, 1.0);

    coherence = sqrt((jxx - jyy).^2 + 4.0 * jxy.^2) ./ ...
                (jxx + jyy + 1e-6);

    structure = min(max(edge .* coherence, 0), 1);

    structureInfo.jxx = jxx;
    structureInfo.jyy = jyy;
    structureInfo.jxy = jxy;
    structureInfo.coherence = coherence;
    structureInfo.structure = structure;
end

%% ========================================================================
%% 子函数 4：噪声分析
%% ========================================================================
function noiseInfo = noise_analyze(Y, edge, texture)
    % 使用 Laplacian 高频响应 + MAD 估计噪声尺度。
    %
    % MAD 比普通标准差更抗真实边缘和纹理干扰。
    % noiseLikelihood 越高，后续锐化增益越低。

    Lap4 = [0 1 0;
            1 -4 1;
            0 1 0];

    lap = conv2_custom(Y, Lap4);

    madVal = median(abs(lap(:) - median(lap(:))));
    noiseSigma = max(1e-3, madVal / 0.6745);

    fineAbs = abs(lap);

    noiseLikelihood = exp(-fineAbs ./ (noiseSigma * 2.5 + 1e-6));

    % 强边缘、强纹理位置不轻易认为是噪声。
    noiseLikelihood = noiseLikelihood .* (1.0 - max(edge, texture * 0.7));
    noiseLikelihood = min(max(noiseLikelihood, 0), 1);

    noiseInfo.lap = lap;
    noiseInfo.noiseSigma = noiseSigma;
    noiseInfo.noiseLikelihood = noiseLikelihood;
end

%% ========================================================================
%% 子函数 5：频率分解
%% ========================================================================
function freq = frequency_decompose(Y)
    % 将 Y 分为三个频段：
    %
    % base：低频亮度基底
    % mid ：中频细节，主要锐化对象
    % fine：高频细节，可能包含纹理，也可能包含噪声
    %
    % Y ≈ base + mid + fine

    base = blur_gaussian(Y, 9, 2.0);
    lowMid = blur_gaussian(Y, 5, 1.0);

    mid = lowMid - base;
    fine = Y - lowMid;

    freq.base = base;
    freq.lowMid = lowMid;
    freq.mid = mid;
    freq.fine = fine;
end

%% ========================================================================
%% 子函数 6：自适应增强
%% ========================================================================
function enhanceInfo = adaptive_enhance(freq, edgeInfo, textureInfo, structureInfo, noiseInfo, cfg)
    % 自适应锐化核心：
    %
    % detail = mid + fineBoost * fine
    %
    % contentGain = max(textureGain, edgeGain)
    %
    % noiseGate = 1 - noiseRejection * noiseLikelihood
    %
    % adaptiveDetail = detail * contentGain * noiseGate * strength
    %
    % 含义：
    % 纹理强的位置多增强；
    % 结构边缘可信的位置适度增强；
    % 噪声可能性高的位置少增强。

    textureGain = cfg.textureBoost * textureInfo.texture;
    edgeGain = cfg.edgeBoost * structureInfo.structure;

    contentGain = max(textureGain, edgeGain);

    noiseGate = 1.0 - cfg.noiseRejection * noiseInfo.noiseLikelihood;

    detail = freq.mid + cfg.fineBoost * freq.fine;

    adaptiveDetail = detail .* contentGain .* noiseGate * cfg.strength;

    candidateY = freq.base + freq.mid + freq.fine + adaptiveDetail;

    enhanceInfo.textureGain = textureGain;
    enhanceInfo.edgeGain = edgeGain;
    enhanceInfo.contentGain = contentGain;
    enhanceInfo.noiseGate = noiseGate;
    enhanceInfo.detail = detail;
    enhanceInfo.adaptiveDetail = adaptiveDetail;
    enhanceInfo.candidateY = candidateY;

    %#ok<NASGU>
    unusedEdge = edgeInfo.edge;
end

%% ========================================================================
%% 子函数 7：伪影控制
%% ========================================================================
function artifactInfo = artifact_control(originalY, candidateY, edgeInfo, textureInfo, cfg)
    % 控制 halo 和局部过冲。
    %
    % halo 风险设计：
    % haloRisk = edge * (1 - texture) * abs(delta)
    %
    % 强边缘 + 低纹理 + 大改变量，最容易出现亮边/暗边。

    delta = candidateY - originalY;
    deltaNorm = normalize01(abs(delta), 98.0);

    haloRisk = edgeInfo.edge .* (1.0 - textureInfo.texture) .* deltaNorm;

    guardedDelta = delta .* (1.0 - cfg.haloGuard * haloRisk);
    guardedY = originalY + guardedDelta;

    [localMin, localMax] = local_min_max(originalY, 5);

    % 纹理区域允许更大的局部变化，平坦区限制更严格。
    margin = cfg.overshootMargin * (1.0 + 2.0 * textureInfo.texture);

    limitedY = min(max(guardedY, localMin - margin), localMax + margin);

    artifactInfo.delta = delta;
    artifactInfo.deltaNorm = deltaNorm;
    artifactInfo.haloRisk = haloRisk;
    artifactInfo.guardedY = guardedY;
    artifactInfo.localMin = localMin;
    artifactInfo.localMax = localMax;
    artifactInfo.margin = margin;
    artifactInfo.limitedY = limitedY;
end

%% ========================================================================
%% 子函数 8：感知重建
%% ========================================================================
function Yout = perceptual_reconstruct(originalY, limitedY, textureInfo, structureInfo, noiseInfo)
    % 最终不是直接使用锐化图，而是和原图做感知加权融合。
    %
    % 纹理/结构越强，越相信锐化结果；
    % 噪声可能性越高，越回退到原图。

    perceptualWeight = min(max(textureInfo.texture + structureInfo.structure, 0), 1);
    perceptualWeight = perceptualWeight .* (1.0 - 0.65 * noiseInfo.noiseLikelihood);

    Yout = originalY .* (1.0 - perceptualWeight) + limitedY .* perceptualWeight;
end

%% ========================================================================
%% 子函数 9：RGB <-> YUV420
%% ========================================================================
function frame = rgb_to_yuv420(rgb)
    src = double(rgb);

    R = src(:, :, 1);
    G = src(:, :, 2);
    B = src(:, :, 3);

    Y = 0.299 * R + 0.587 * G + 0.114 * B;
    Ufull = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0;
    Vfull = 0.5 * R - 0.418688 * G - 0.081312 * B + 128.0;

    [h, w] = size(Y);

    padH = mod(h, 2);
    padW = mod(w, 2);

    Ufull = pad_bottom_right(Ufull, padH, padW);
    Vfull = pad_bottom_right(Vfull, padH, padW);

    U = (Ufull(1:2:end, 1:2:end) + Ufull(2:2:end, 1:2:end) + ...
         Ufull(1:2:end, 2:2:end) + Ufull(2:2:end, 2:2:end)) * 0.25;

    V = (Vfull(1:2:end, 1:2:end) + Vfull(2:2:end, 1:2:end) + ...
         Vfull(1:2:end, 2:2:end) + Vfull(2:2:end, 2:2:end)) * 0.25;

    frame.y = Y;
    frame.u = U;
    frame.v = V;
    frame.height = h;
    frame.width = w;
end

function rgb = yuv420_to_rgb(frame)
    Y = double(frame.y);

    U = kron(double(frame.u), ones(2, 2));
    V = kron(double(frame.v), ones(2, 2));

    U = U(1:frame.height, 1:frame.width);
    V = V(1:frame.height, 1:frame.width);

    UU = U - 128.0;
    VV = V - 128.0;

    R = Y + 1.402 * VV;
    G = Y - 0.344136 * UU - 0.714136 * VV;
    B = Y + 1.772 * UU;

    rgb = uint8(min(max(round(cat(3, R, G, B)), 0), 255));
end

%% ========================================================================
%% 子函数 10：自定义卷积、滤波、局部统计
%% ========================================================================
function out = conv2_custom(img, kernel)
    [kh, kw] = size(kernel);

    py = floor(kh / 2);
    px = floor(kw / 2);

    padded = pad_reflect(img, py, px);

    out = conv2(padded, kernel, 'valid');
end

function out = blur_gaussian(img, kernelSize, sigma)
    kernel = gaussian_kernel(kernelSize, sigma);
    out = conv2_custom(img, kernel);
end

function kernel = gaussian_kernel(kernelSize, sigma)
    radius = floor(kernelSize / 2);

    [xx, yy] = meshgrid(-radius:radius, -radius:radius);

    kernel = exp(-(xx.^2 + yy.^2) / (2.0 * sigma * sigma));
    kernel = kernel / sum(kernel(:));
end

function out = box_mean(img, kernelSize)
    kernel = ones(kernelSize, kernelSize) / (kernelSize * kernelSize);
    out = conv2_custom(img, kernel);
end

function [localMin, localMax] = local_min_max(img, kernelSize)
    radius = floor(kernelSize / 2);

    padded = pad_reflect(img, radius, radius);

    [h, w] = size(img);

    localMin = zeros(h, w);
    localMax = zeros(h, w);

    for row = 1:h
        for col = 1:w
            block = padded(row:row + kernelSize - 1, col:col + kernelSize - 1);
            localMin(row, col) = min(block(:));
            localMax(row, col) = max(block(:));
        end
    end
end

%% ========================================================================
%% 子函数 11：归一化与 padding
%% ========================================================================
function y = normalize01(x, percentileValue)
    scale = prctile(abs(x(:)), percentileValue);

    if scale < 1e-6
        y = zeros(size(x));
    else
        y = x / scale;
        y = min(max(y, 0), 1);
    end
end

function padded = pad_reflect(img, py, px)
    [h, w] = size(img);

    rowIdx = reflect_indices(1 - py:h + py, h);
    colIdx = reflect_indices(1 - px:w + px, w);

    padded = img(rowIdx, colIdx);
end

function idx = reflect_indices(idx, n)
    if n == 1
        idx = ones(size(idx));
        return;
    end

    period = 2 * n - 2;

    idx = mod(idx - 1, period) + 1;

    over = idx > n;
    idx(over) = period - idx(over) + 2;
end

function out = pad_bottom_right(img, padH, padW)
    out = img;

    if padH > 0
        out = [out; out(end, :)];
    end

    if padW > 0
        out = [out, out(:, end)];
    end
end

%% ========================================================================
%% 子函数 12：保存图像
%% ========================================================================
function save_debug(debugDir, name, data)
    if ndims(data) == 3
        img = uint8(min(max(round(data), 0), 255));
    else
        img = to_uint8_gray(data);
    end

    imwrite(img, fullfile(debugDir, name));
end

function img = to_uint8_gray(data)
    data = double(data);
    data(~isfinite(data)) = 0;

    low = prctile(data(:), 1);
    high = prctile(data(:), 99);

    if high - low < 1e-6
        low = min(data(:));
        high = max(data(:));
    end

    if high - low < 1e-6
        img = uint8(zeros(size(data)));
    else
        img = uint8(min(max(round((data - low) / (high - low) * 255), 0), 255));
    end
end

function ensure_parent_dir(filePath)
    parentDir = fileparts(filePath);

    if ~isempty(parentDir) && ~exist(parentDir, 'dir')
        mkdir(parentDir);
    end
end
