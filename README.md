function [Yout, info] = auto_dark_tone_y(Y, blackPct, whitePct, gammaValue, targetP90, maxGain)
    % 面向极暗 8bit 图的自动暗光 tone 模块
    %
    % 作用：
    % 1. 用百分位黑白点拉开动态范围
    % 2. 用 gamma 提亮暗部
    % 3. 把 p90 推到目标亮度
    % 4. 限制最大增益，避免噪声无限放大
    %
    % 推荐：
    % [Y_tone, info] = auto_dark_tone_y(Y, 0.5, 99.5, 0.55, 180, 8.0);

    Y = double(Y);

    black = prctile(Y(:), blackPct);
    white = prctile(Y(:), whitePct);

    if white - black < 5
        white = black + 5;
    end

    % 1. 百分位动态范围拉伸
    X = (Y - black) / (white - black);
    X = min(max(X, 0), 1);

    % 2. gamma 提亮
    Xg = X .^ gammaValue;

    % 3. 自动曝光：把 p90 推到 targetP90
    p90 = prctile(Xg(:), 90);
    target = targetP90 / 255.0;

    gain = target / max(p90, 1e-4);
    gain = min(max(gain, 1.0), maxGain);

    Xe = Xg * gain;

    % 4. 高光 shoulder，防止拉伸后高光直接炸白
    shoulder = 0.8;
    Xe = Xe ./ (1.0 + shoulder * Xe);

    % 归一化，让 1 仍接近 1
    whiteNorm = gain / (1.0 + shoulder * gain);
    Xe = Xe / max(whiteNorm, 1e-6);

    Yout = min(max(Xe, 0), 1) * 255.0;

    info.black = black;
    info.white = white;
    info.gain = gain;
    info.X = X;
    info.Xg = Xg;
    info.Yout = Yout;
    info.delta = Yout - Y;
end
