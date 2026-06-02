function [rgbOut, info] = adaptive_dark_scurve_rgb(rgbIn)
    % 自适应暗光 S 曲线 tone mapping，RGB 输入版本
    %
    % 输入：
    %   rgbIn: 0~1 double RGB
    %
    % 输出：
    %   rgbOut: 0~1 double RGB
    %
    % 设计目标：
    %   1. 暗区明显提亮
    %   2. S 曲线 pivot 跟随图像直方图，而不是固定 0.5
    %   3. 暗部幂函数只作用在暗区
    %   4. 保护相对亮区 p70~p95 的纹理细节
    %   5. 通过亮度比例缩放 RGB，尽量避免偏色

    rgb = min(max(double(rgbIn), 0), 1);

    R = rgb(:, :, 1);
    G = rgb(:, :, 2);
    B = rgb(:, :, 3);

    % 使用 BT.601 亮度作为 tone 控制通道
    Yin = 0.299 * R + 0.587 * G + 0.114 * B;
    Yin = min(max(Yin, 0), 1);

    % 统计当前图像主体亮度分布
    p01 = prctile(Yin(:), 1);
    p10 = prctile(Yin(:), 10);
    p40 = prctile(Yin(:), 40);
    p50 = prctile(Yin(:), 50);
    p70 = prctile(Yin(:), 70);
    p90 = prctile(Yin(:), 90);
    p95 = prctile(Yin(:), 95);
    p99 = prctile(Yin(:), 99);

    % S 曲线中点跟随暗图主体范围。
    % 你的例子 p10=0.052, p90=0.408，则 pivot 约 0.23。
    pivot = 0.5 * (p10 + p90);

    % slope 不要太大，否则 p70~p95 斜率容易被压低，亮区纹理会丢。
    slope = 3.2;

    S = 1 ./ (1 + exp(-slope * (Yin - pivot)));

    S0 = 1 / (1 + exp(-slope * (0 - pivot)));
    S1 = 1 / (1 + exp(-slope * (1 - pivot)));

    Ysc = (S - S0) / max(S1 - S0, 1e-6);
    Ysc = min(max(Ysc, 0), 1);

    % 暗部幂函数增强。
    % gammaDark < 1 会抬暗部。
    gammaDark = 0.62;
    Ypow = Yin .^ gammaDark;

    % 暗部 mask 只覆盖暗区和中暗区。
    % 不要让它影响 p70~p90，否则相对亮区纹理容易被压平。
    darkMask = 1.0 - smoothstep(p40, p75_safe(p70, p90), Yin);
    darkMask = min(max(darkMask, 0), 1);

    % 黑位保护，避免接近 0 的区域被直接抬成灰雾。
    blackProtect = smoothstep(max(p01, 0.002), max(p10, 0.01), Yin);
    darkMask = darkMask .* blackProtect;

    % 混合 S 曲线和暗部幂函数。
    % darkPowerAmount 越大，暗区越亮。
    darkPowerAmount = 0.70;
    Ytone = Ysc .* (1 - darkPowerAmount * darkMask) + ...
            Ypow .* (darkPowerAmount * darkMask);

    % 保护相对亮区的纹理细节。
    % 你的 p90 只有 0.408，所以这里的“亮区”不是 0.8 以上，
    % 而是当前图像分布里的 p70~p95。
    brightMask = smoothstep(p70, p95, Yin);

    % 线性参考用于保留相对亮区的亮度层次。
    Ylinear = (Yin - p01) / max(p99 - p01, 1e-6);
    Ylinear = min(max(Ylinear, 0), 1);

    brightPreserveAmount = 0.28;
    Ytone = Ytone .* (1 - brightPreserveAmount * brightMask) + ...
            Ylinear .* (brightPreserveAmount * brightMask);

    % 再加一点亮区局部细节回灌，修复 tone 后亮区纹理变糊的问题。
    base = blur_gaussian(Yin, 15, 3.0);
    detail = Yin - base;

    detailRestoreAmount = 0.22;
    Ytone = Ytone + detailRestoreAmount * brightMask .* detail;

    Ytone = min(max(Ytone, 0), 1);

    % 用亮度比例缩放 RGB，避免 R/G/B 分别 tone 造成偏色。
    ratio = Ytone ./ max(Yin, 1e-5);

    % 限制 ratio，避免极暗像素被无限放大导致彩噪/色斑。
    ratio = min(max(ratio, 0.25), 8.0);

    rgbOut = rgb .* ratio;
    rgbOut = min(max(rgbOut, 0), 1);

    info.Yin = Yin;
    info.Ysc = Ysc;
    info.Ypow = Ypow;
    info.Ytone = Ytone;
    info.ratio = ratio;
    info.darkMask = darkMask;
    info.brightMask = brightMask;
    info.detail = detail;
    info.p01 = p01;
    info.p10 = p10;
    info.p40 = p40;
    info.p50 = p50;
    info.p70 = p70;
    info.p90 = p90;
    info.p95 = p95;
    info.p99 = p99;
    info.pivot = pivot;
    info.slope = slope;
end


function y = smoothstep(edge0, edge1, x)
    t = (x - edge0) / max(edge1 - edge0, 1e-6);
    t = min(max(t, 0), 1);
    y = t .* t .* (3.0 - 2.0 * t);
end

function p75 = p75_safe(p70, p90)
    % 给暗部幂函数 mask 一个安全结束点。
    % 不让暗部增强拖到 p90，保护相对亮区纹理。
    p75 = p70 + 0.35 * (p90 - p70);
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

function out = conv2_custom(img, kernel)
    [kh, kw] = size(kernel);
    py = floor(kh / 2);
    px = floor(kw / 2);
    padded = pad_reflect(img, py, px);
    out = conv2(padded, kernel, 'valid');
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
