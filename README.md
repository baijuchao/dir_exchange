function [Yout, info] = local_hist_contrast_y(Y, tileRows, tileCols, clipLimit, maxGain)
    % Y 域局部直方图增强 / 简化 CLAHE
    %
    % Y:
    %   输入亮度，范围 0~255
    %
    % tileRows, tileCols:
    %   tile 数量，比如 8, 8
    %
    % clipLimit:
    %   直方图裁剪强度，建议 0.003 ~ 0.015
    %
    % maxGain:
    %   最大亮度增益，建议 1.6 ~ 2.5
    %
    % 输出：
    %   Yout: 增强后的亮度
    %   info: 调试信息

    Y = double(Y);
    [h, w] = size(Y);

    Yu8 = uint8(min(max(round(Y), 0), 255));

    % 每个 tile 的尺寸
    tileH = ceil(h / tileRows);
    tileW = ceil(w / tileCols);

    % 保存每个 tile 的 LUT，大小：tileRows x tileCols x 256
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

    % 对每个像素，根据所在 tile 周围 4 个 LUT 做双线性插值
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

    % gain 限幅，避免暗部被局部直方图暴力拉亮
    gain = Yeq ./ max(Y, 1.0);
    gain = min(max(gain, 1.0 / maxGain), maxGain);

    Yout = Y .* gain;
    Yout = min(max(Yout, 0), 255);

    info.Yeq = Yeq;
    info.gain = gain;
    info.delta = Yout - Y;
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

    % clipLimit 是归一化比例，转成每个 bin 的最大计数
    clipCount = max(1, clipLimit * numPixels);

    excess = sum(max(histVals - clipCount, 0));
    histVals = min(histVals, clipCount);

    % 把被裁剪掉的计数平均分回所有 bin
    histVals = histVals + excess / 256;

    cdf = cumsum(histVals);
    cdf = cdf / cdf(end);

    lut = cdf * 255;
end
