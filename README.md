function [rgbOut, info] = advanced_contrast_rgb(rgbIn)
    % 高级亮度域对比度增强
    %
    % 输入:
    %   rgbIn: 0~1 double RGB
    %
    % 输出:
    %   rgbOut: 0~1 double RGB
    %
    % 核心:
    %   多尺度 CLAHE 只作用在亮度 Y 上，再用 ratio 回乘 RGB。

    rgb = min(max(double(rgbIn), 0), 1);

    R = rgb(:, :, 1);
    G = rgb(:, :, 2);
    B = rgb(:, :, 3);

    Y = 0.299 * R + 0.587 * G + 0.114 * B;
    Y = min(max(Y, 0), 1);

    % 多尺度 CLAHE
    % 小 tile 增强局部细节，大 tile 增强大尺度层次。
    [YclaheSmall, infoSmall] = clahe_y_custom(Y, 8, 8, 0.006);
    [YclaheLarge, infoLarge] = clahe_y_custom(Y, 16, 16, 0.004);

    % 混合两个尺度
    Yclahe = 0.65 * YclaheSmall + 0.35 * YclaheLarge;

    % 避免 CLAHE 过度改变高光，加入亮区保护
    highlightMask = smoothstep(0.75, 0.98, Y);

    % 高光区域更多保留原始 Y
    highlightProtectAmount = 0.45;
    Ytarget = Yclahe .* (1 - highlightProtectAmount * highlightMask) + ...
              Y .* (highlightProtectAmount * highlightMask);

    % 暗区不要无限拉，避免噪声爆
    ratio = Ytarget ./ max(Y, 1e-5);

    % 根据亮度自适应限制增益：
    % 暗部允许较大增益，中间调适中，高光接近 1。
    maxGainShadow = 3.0;
    maxGainMid = 1.8;
    maxGainHigh = 1.15;

    shadowMask = 1.0 - smoothstep(0.10, 0.35, Y);
    highMask = smoothstep(0.70, 0.95, Y);
    midMask = 1.0 - shadowMask - highMask;
    midMask = min(max(midMask, 0), 1);

    maxGainMap = maxGainShadow * shadowMask + ...
                 maxGainMid * midMask + ...
                 maxGainHigh * highMask;

    ratio = min(max(ratio, 0.5), maxGainMap);

    rgbOut = rgb .* ratio;
    rgbOut = min(max(rgbOut, 0), 1);

    info.Y = Y;
    info.YclaheSmall = YclaheSmall;
    info.YclaheLarge = YclaheLarge;
    info.Yclahe = Yclahe;
    info.Ytarget = Ytarget;
    info.ratio = ratio;
    info.highlightMask = highlightMask;
    info.maxGainMap = maxGainMap;
    info.infoSmall = infoSmall;
    info.infoLarge = infoLarge;
end


function [Yout, info] = clahe_y_custom(Y, tileRows, tileCols, clipLimit)
    % 自定义 Y 域 CLAHE，输入输出 0~1

    Y = min(max(double(Y), 0), 1);
    [h, w] = size(Y);

    Yu8 = uint8(round(Y * 255));

    tileH = ceil(h / tileRows);
    tileW = ceil(w / tileCols);

    LUT = zeros(tileRows, tileCols, 256);

    for tr = 1:tileRows
        for tc = 1:tileCols
            r1 = (tr - 1) * tileH + 1;
            r2 = min(tr * tileH, h);

            c1 = (tc - 1) * tileW + 1;
            c2 = min(tc * tileW, w);

            tile = Yu8(r1:r2, c1:c2);
            LUT(tr, tc, :) = build_clahe_lut(tile, clipLimit);
        end
    end

    Yeq = zeros(h, w);

    for r = 1:h
        tileY = (r - 0.5) / tileH + 0.5;
        tr0 = floor(tileY);
        tr1 = tr0 + 1;
        wy = tileY - tr0;

        tr0 = min(max(tr0, 1), tileRows);
        tr1 = min(max(tr1, 1), tileRows);

        for c = 1:w
            tileX = (c - 0.5) / tileW + 0.5;
            tc0 = floor(tileX);
            tc1 = tc0 + 1;
            wx = tileX - tc0;

            tc0 = min(max(tc0, 1), tileCols);
            tc1 = min(max(tc1, 1), tileCols);

            bin = double(Yu8(r, c)) + 1;

            v00 = LUT(tr0, tc0, bin);
            v01 = LUT(tr0, tc1, bin);
            v10 = LUT(tr1, tc0, bin);
            v11 = LUT(tr1, tc1, bin);

            top = (1 - wx) * v00 + wx * v01;
            bot = (1 - wx) * v10 + wx * v11;

            Yeq(r, c) = (1 - wy) * top + wy * bot;
        end
    end

    Yout = min(max(Yeq / 255.0, 0), 1);

    info.tileRows = tileRows;
    info.tileCols = tileCols;
    info.clipLimit = clipLimit;
    info.Yeq = Yout;
end


function lut = build_clahe_lut(tile, clipLimit)
    % 为单个 tile 构建 CLAHE LUT

    tile = double(tile(:));
    numPixels = numel(tile);

    histVals = zeros(256, 1);

    for i = 1:numPixels
        bin = tile(i) + 1;
        histVals(bin) = histVals(bin) + 1;
    end

    % clipLimit 是相对比例，转成每个 bin 最大计数
    clipCount = max(1, clipLimit * numPixels);

    excess = sum(max(histVals - clipCount, 0));
    histVals = min(histVals, clipCount);

    % 裁剪掉的数量均匀分回所有 bin
    histVals = histVals + excess / 256;

    cdf = cumsum(histVals);
    cdf = cdf / max(cdf(end), 1e-6);

    lut = cdf * 255;
end

function y = smoothstep(edge0, edge1, x)
    t = (x - edge0) / max(edge1 - edge0, 1e-6);
    t = min(max(t, 0), 1);
    y = t .* t .* (3.0 - 2.0 * t);
end
