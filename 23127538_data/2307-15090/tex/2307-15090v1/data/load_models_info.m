function [path, peak] = load_models_info(info)
models_info = load('./vals/models_info.mat').models_info;
path_key = [info, '_pth'];
peak_key = [info, '_peak'];
path = models_info.(path_key);
peak = models_info.(peak_key);
end
