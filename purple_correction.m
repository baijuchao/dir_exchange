1. 找紫色异常候选
2. 找污染源：高亮、小球、强反差亮物体
3. 根据污染源距离生成 source_conf
4. 用连通域判断哪些紫色区域和污染源有关
5. 保护真实紫色大区域
6. 只修 Cb/Cr 色度，不动 Y
7. 平滑修复量，不平滑图像
function [rgbOut, info] = defringe_purple_pollution_rgb(rgbIn, opts)
% 非神经网络紫色污染/紫边去除
% 输入:
%   rgbIn: double/single RGB, range 0~1
%
% 核心思想:
%   不再用“紫色mask + 边缘mask”。
%   改成“紫色异常 + 污染源距离场 + 连通域 + 真实紫色保护”。

    if nargin < 2
        opts = struct();
    end

    opts = fill_default_opts(opts);

    rgb = min(max(double(rgbIn), 0), 1);

    R = rgb(:,:,1);
    G = rgb(:,:,2);
    B = rgb(:,:,3);

    Y  = 0.299 * R + 0.587 * G + 0.114 * B;
    Cb = (B - Y) / 1.772;
    Cr = (R - Y) / 1.402;

    %% 1. 紫色/洋红异常候选
    hsvImg = rgb2hsv(rgb);
    H = hsvImg(:,:,1);
    S = hsvImg(:,:,2);

    magenta_excess = max(0, 0.5 * (R + B) - G);

    hue_purple = (H >= opts.hueLow1 & H <= opts.hueHigh1) | ...
                 (H >= opts.hueLow2 & H <= opts.hueHigh2);

    purple_candidate = hue_purple & ...
                       S > opts.minSat & ...
                       B > G + opts.bgMargin & ...
                       R > G + opts.rgMargin & ...
                       magenta_excess > opts.magentaTh;

    purple_strength = magenta_excess ./ (magenta_excess + opts.magentaK);
    purple_strength = min(max(purple_strength, 0), 1);

    %% 2. 找污染源：高亮源 + 强反差亮物体
    win = true(opts.localWin);
    Ymax = ordfilt2(Y, numel(win), win);

    highlight_source = Ymax > opts.highlightTh;

    edge_band = abs(imgaussfilt(Y, opts.edgeSigmaSmall) - ...
                    imgaussfilt(Y, opts.edgeSigmaLarge));

    contrast_source = edge_band > opts.edgeTh;

    % 亮物体更容易成为紫色污染源，避免暗部纹理误触发
    bright_context = Ymax > opts.brightContextTh;

    source = highlight_source | (contrast_source & bright_context);

    %% 3. 污染源距离场，不再依赖边缘线
    if any(source(:))
        dist_to_source = bwdist(source);
        source_conf = exp(-(dist_to_source / opts.sourceRadius).^2);
    else
        dist_to_source = inf(size(Y));
        source_conf = zeros(size(Y));
    end

    source_zone = source_conf > opts.sourceConfMin;

    %% 4. 连通域判断：紫色区域是否和污染源有关
    CC = bwconncomp(purple_candidate, 8);
    pollution_region = false(size(Y));

    for i = 1:CC.NumObjects
        pix = CC.PixelIdxList{i};

        area_i = numel(pix);
        mean_source_conf = mean(source_conf(pix));
        max_source_conf = max(source_conf(pix));
        mean_magenta = mean(magenta_excess(pix));
        mean_sat = mean(S(pix));

        near_source = max_source_conf > opts.compMaxSourceConfTh || ...
                      mean_source_conf > opts.compMeanSourceConfTh;

        chroma_abnormal = mean_magenta > opts.compMagentaTh && ...
                          mean_sat > opts.compSatTh;

        not_too_large_far_region = ~(area_i > opts.realPurpleAreaTh && ...
                                     mean_source_conf < opts.realPurpleFarSourceTh);

        if near_source && chroma_abnormal && not_too_large_far_region
            pollution_region(pix) = true;
        end
    end

    %% 5. 真实紫色保护
    % 真实紫色常见特征:
    % - 面积大
    % - 离高亮/强反差源较远
    % - 色度连续稳定
    large_purple = bwareaopen(purple_candidate, opts.realPurpleAreaTh);

    chroma = sqrt(Cb.^2 + Cr.^2);
    chroma_smooth = imgaussfilt(chroma, opts.chromaStableSigma);
    chroma_var = imgaussfilt((chroma - chroma_smooth).^2, opts.chromaStableSigma);

    stable_chroma = 1 ./ (1 + chroma_var / opts.chromaVarK);

    far_from_source = source_conf < opts.realPurpleFarSourceTh;

    real_purple_protect = double(large_purple) .* ...
                          double(far_from_source) .* ...
                          stable_chroma;

    real_purple_protect = min(max(real_purple_protect, 0), 1);

    %% 6. 最终污染置信度
    conf = double(pollution_region) .* ...
           purple_strength .* ...
           source_conf;

    conf = conf .* (1 - opts.realPurpleProtectAmount * real_purple_protect);

    % 轻微扩散，覆盖紫色 halo 的软边界
    conf = imgaussfilt(conf, opts.confSmoothSigma);
    conf = min(max(conf, 0), 1);

    %% 7. 参考色度估计
    % 参考像素：不是紫色污染、不是明显紫色候选、不过曝
    ref_valid = conf < opts.refConfMax & ...
                ~purple_candidate & ...
                Y < opts.refYMax;

    ref_w = double(ref_valid);

    denom = imgaussfilt(ref_w, opts.refSigma) + 1e-6;

    Cb_ref = imgaussfilt(Cb .* ref_w, opts.refSigma) ./ denom;
    Cr_ref = imgaussfilt(Cr .* ref_w, opts.refSigma) ./ denom;

    %% 8. 修复量传播：平滑 dCb/dCr，不平滑图像
    dCb = Cb_ref - Cb;
    dCr = Cr_ref - Cr;

    fixDen = imgaussfilt(conf, opts.fixSigma) + 1e-6;

    dCb_s = imgaussfilt(dCb .* conf, opts.fixSigma) ./ fixDen;
    dCr_s = imgaussfilt(dCr .* conf, opts.fixSigma) ./ fixDen;

    Cb_new = Cb + opts.amount * conf .* dCb_s;
    Cr_new = Cr + opts.amount * conf .* dCr_s;

    %% 9. YCbCr 回 RGB，只改色度，Y 不动
    R2 = Y + 1.402 * Cr_new;
    B2 = Y + 1.772 * Cb_new;
    G2 = (Y - 0.299 * R2 - 0.114 * B2) / 0.587;

    rgbOut = cat(3, R2, G2, B2);
    rgbOut = min(max(rgbOut, 0), 1);

    %% Debug info
    info.Y = Y;
    info.magenta_excess = magenta_excess;
    info.purple_candidate = purple_candidate;
    info.highlight_source = highlight_source;
    info.contrast_source = contrast_source;
    info.source = source;
    info.source_conf = source_conf;
    info.source_zone = source_zone;
    info.pollution_region = pollution_region;
    info.real_purple_protect = real_purple_protect;
    info.conf = conf;
    info.Cb_ref = Cb_ref;
    info.Cr_ref = Cr_ref;
    info.dCb_s = dCb_s;
    info.dCr_s = dCr_s;
end
