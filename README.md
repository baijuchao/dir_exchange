function [rgbOut, info] = auto_exposure_rgb_double(rgbIn, targetP90, maxGain, blackPoint)
    % 线性 RGB double 自动曝光
    %
    % rgbIn:
    %   线性 RGB，范围建议 0~1
    %
    % targetP90:
    %   目标 p90 亮度，比如 0.35~0.65
    %
    % maxGain:
    %   最大数字增益，比如 4~16
    %
    % blackPoint:
    %   黑位，比如 0.0~0.02

    rgb = double(rgbIn);

    R = rgb(:, :, 1);
    G = rgb(:, :, 2);
    B = rgb(:, :, 3);

    Y = 0.299 * R + 0.587 * G + 0.114 * B;

    Ycorr = max(Y - blackPoint, 0);

    p90 = prctile(Ycorr(:), 90);

    gain = targetP90 / max(p90, 1e-8);
    gain = min(max(gain, 1.0), maxGain);

    rgbOut = max(rgb - blackPoint, 0) * gain + blackPoint;

    % 线性域暂时允许超过 1，后面的 tone curve 可以压高光。
    % 如果你没有高光压缩模块，也可以 clamp 到 1。
    rgbOut = max(rgbOut, 0);

    info.p90 = p90;
    info.gain = gain;
    info.blackPoint = blackPoint;
    info.targetP90 = targetP90;
end
