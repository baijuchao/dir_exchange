function Yout = adaptive_dark_scurve(Y)
    Y = double(Y);
    Yn = min(max(Y, 0), 1);

    p10 = prctile(Yn(:), 10);
    p50 = prctile(Yn(:), 50);
    p70 = prctile(Yn(:), 70);
    p90 = prctile(Yn(:), 90);

    % S 曲线中点跟随图像主体，不固定 0.5
    pivot = 0.5 * (p10 + p90);
    slope = 3.8;

    S = 1 ./ (1 + exp(-slope * (Yn - pivot)));

    S0 = 1 / (1 + exp(-slope * (0 - pivot)));
    S1 = 1 / (1 + exp(-slope * (1 - pivot)));

    Ysc = (S - S0) / (S1 - S0);
    Ysc = min(max(Ysc, 0), 1);

    % 暗部幂函数增强，只作用于 p10~p70 以下
    gammaDark = 0.65;
    Ypow = Yn .^ gammaDark;

    darkMask = 1.0 - smoothstep(p50, p90, Yn);
    darkMask = min(max(darkMask, 0), 1);

    % p90 附近尽量回到 S 曲线，不让暗部增强影响相对亮区
    Ymix = Ysc .* (1 - darkMask) + Ypow .* darkMask;

    % 相对亮区保护：p80 以上尽量保留层次，不继续猛拉
    highlightProtect = smoothstep(p70, p90, Yn);
    Yout = Ymix .* (1 - 0.35 * highlightProtect) + Ysc .* (0.35 * highlightProtect);

    Yout = min(max(Yout, 0), 1);
end


function y = smoothstep(edge0, edge1, x)
    t = (x - edge0) / max(edge1 - edge0, 1e-6);
    t = min(max(t, 0), 1);
    y = t .* t .* (3.0 - 2.0 * t);
end
