function [Yout, info] = dark_region_lift_y(Y, liftStrength, shadowEnd, blackPoint, maxAddCode)
    % 专门的 Y 域暗区提亮模块
    %
    % Y:
    %   输入亮度，0~255
    %
    % liftStrength:
    %   提亮强度，建议 0.8~2.0
    %
    % shadowEnd:
    %   暗区结束位置，0~1，例如 0.65 表示 65% 以下都可被提亮
    %
    % blackPoint:
    %   黑位保护，0~1，例如 0.02~0.08
    %
    % maxAddCode:
    %   每个像素最大加亮码值，建议 60~140
    %
    % 推荐第一组：
    % [Y_lift, info] = dark_region_lift_y(Y, 1.35, 0.75, 0.03, 110);

    Y = double(Y);
    Yn = Y / 255.0;

    % 1. 暗区权重：
    %    低亮度位置权重大，高于 shadowEnd 后逐渐变 0。
    shadowWeight = 1.0 - smoothstep(blackPoint, shadowEnd, Yn);
    shadowWeight = min(max(shadowWeight, 0), 1);

    % 2. 黑位保护：
    %    纯黑附近不直接抬，避免黑底变灰。
    blackProtect = smoothstep(0.0, blackPoint, Yn);
    blackProtect = min(max(blackProtect, 0), 1);

    % 3. 强暗部 lift 曲线：
    %    log 曲线在暗部增量更明显，比 gamma/sqrt 更直接。
    %
    %    liftCurveNorm 范围大约 0~1，暗部贡献更强。
    liftCurveNorm = log(1.0 + liftStrength * (1.0 - Yn)) / log(1.0 + liftStrength);

    % 4. 暗部基础加亮量：
    %    越暗，加得越多；接近 shadowEnd 时被 shadowWeight 压下去。
    delta = maxAddCode * liftCurveNorm .* shadowWeight .* blackProtect;

    % 5. 可选：暗区纹理保护。
    %    有细节的暗区多提，极平坦暗区略少提，避免噪声和黑底浮起来。
    localMean = box_mean(Yn, 7);
    localMeanSq = box_mean(Yn.^2, 7);
    localVar = max(localMeanSq - localMean.^2, 0);
    textureLike = normalize01(sqrt(localVar), 98.0);

    textureGate = 0.65 + 0.35 * textureLike;
    delta = delta .* textureGate;

    % 6. 应用提亮。
    Yout = Y + delta;
    Yout = min(max(Yout, 0), 255);

    info.shadowWeight = shadowWeight;
    info.blackProtect = blackProtect;
    info.liftCurveNorm = liftCurveNorm;
    info.textureLike = textureLike;
    info.textureGate = textureGate;
    info.delta = Yout - Y;
    info.Yout = Yout;
end
