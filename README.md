function [Yout, info] = local_hist_contrast_y_shadow(Y, tileRows, tileCols, clipLimit, maxAddCode, maxGain)
    % 暗区优先的 Y 域局部直方图增强
    %
    % 目标：
    % 1. 暗区和中暗区明显提亮
    % 2. 亮区和高光尽量不动
    % 3. 避免 CLAHE 把亮区增强得太夸张
    %
    % 推荐参数：
    % [Yc, info] = local_hist_contrast_y_shadow(Y, 8, 8, 0.006, 60, 2.4);

    Y = double(Y);
    [h, w] = size(Y);

    Yu8 = uint8(min(max(round(Y), 0), 255));

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

    % 先得到普通 CLAHE 的映射结果
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

    Yn = Y / 255.0;

    % 暗区权重：
    % 0~0.45 主要作用，0.45~0.75 逐渐减弱，亮区基本不动。
    shadowWeight = 1.0 - smoothstep(0.45, 0.75, Yn);

    % 黑位保护：
    % 避免纯黑、近黑被直接抬成灰雾。
    blackProtect = smoothstep(0.015, 0.06, Yn);

    % 高光保护：
    % 亮区不允许被 CLAHE 明显增强。
    highlightProtect = 1.0 - smoothstep(0.70, 0.95, Yn);

    weight = shadowWeight .* blackProtect .* highlightProtect;
    weight = min(max(weight, 0), 1);

    % 只使用 CLAHE 的正向提亮，不让它压暗暗区。
    claheDelta = max(Yeq - Y, 0);

    % 如果局部直方图没有明显抬暗区，用一个暗部 lift 作为兜底。
    % sqrt(Y)-Y 在暗部增量最大，高亮区自然变小。
    liftCurve = (sqrt(max(Yn, 0)) - Yn) * 255.0;

    % 暗区提亮量：CLAHE 和 lift 取较强者。
    delta = max(claheDelta, 0.75 * liftCurve);

    % 只在暗区混合，亮区被 weight 压住。
    delta = delta .* weight;

    % 限制最大加亮幅度，避免暗部噪声被拉爆。
    delta = min(delta, maxAddCode);

    candidate = Y + delta;

    % 再限制最大增益。暗像素可以抬，但不能无限放大。
    candidate = min(candidate, max(Y * maxGain, Y + 1.0));

    Yout = min(max(candidate, 0), 255);

    info.Yeq = Yeq;
    info.shadowWeight = shadowWeight;
    info.blackProtect = blackProtect;
    info.highlightProtect = highlightProtect;
    info.weight = weight;
    info.claheDelta = claheDelta;
    info.liftCurve = liftCurve;
    info.delta = Yout - Y;
end
