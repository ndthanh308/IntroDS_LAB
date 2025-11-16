function  data2txt(opt, varargin) 
% Takes matlab variables (vectors and matrices) in varargin and stores it with labels in text file 
% "fname.txt"  to be read by tikz plot. 
%In: 
% opt. 
%         fname    string      string filename without ".txt" ending | 
%                              default = 'data2txtOUT' 
%         (efill   string      filling for empty slots in text file | default = ' ') 
%         (ndata   1x1 int    maximum number of datapoints | default = 200) 
%         minval   1 x 1      minimum value (default ist -1e5) 
%         maxval   1 x 1       maximum value (default ist 1e5) 
%         var_names cell arry of length nargin-1 
% varargin 
%         a         n x 1       vectors in file stored with label 'a' 
%         A         n x m       matrix columns are labeled with _1..._m 
%---------------------------------------------------------------------------- 
% Last edited: Jonas Umlauft, 09/2018 
if isfield(opt,'ndata'), min_datapoints = opt.ndata; 
else min_datapoints = 200; end 
 
if isfield(opt,'fname'), filename = opt.fname; 
else 
    warning('Add field options.fname, now saved as "data2txtOUT.txt" '); 
    filename = 'data2txtOUT'; 
end 
 
if isfield(opt,'efill'), emptyfill = opt.efill; 
else emptyfill = ' ';  end 
 
if isfield(opt,'efill'), emptyfill = opt.efill; 
else emptyfill = ' ';  end 
 
if isfield(opt,'var_names'), var_names = opt.var_names; 
else, for n=2:nargin, var_names{n-1} = inputname(n);end; end 
 
if ~isfield(opt,'minval'),  opt.minval = -1e5; end 
if ~isfield(opt,'maxval'),  opt.maxval = 1e5; end 
 
%% Surface Plot 
% if isfield(options,'surf') && options.surf 
%     if length(varargin) ==3 
%         x = varargin{1}; y = varargin{2}; z = varargin{3}; 
%         printstr = '%f  %f  %f'; 
%         header = 'x y z'; 
%         fileID = fopen([filename '.txt'],'w'); 
%         fprintf(fileID,header); 
%         fprintf(fileID,'\n'); 
%         for j=1:numel(y) 
%             for i=1:numel(x) 
%                 fprintf(fileID,printstr,[x(i) y(j) z(i,j)]); 
%                 fprintf(fileID,'\n'); 
%             end 
%             fprintf(fileID,'\n'); 
%         end 
%  
%         fclose(fileID); 
%  
%     else 
%         error('Not exactly 3 input args for surface plot'); 
%     end 
% return;  
% end 
 
 
 
%% Regular Plot 
header = {}; 
printstr = []; 
data = []; 
notNAN = []; 
for i=1:nargin-1 
    data_i = varargin{i}; 
    if iscell(data_i) 
        data_ic = squeeze(data_i); 
        if size(data_ic,1) ==1, data_ic = data_ic'; end 
        if size(data_ic,2) ~=1, warning('only 1D cells are handeled'); end 
        for c=1:length(data_ic) 
            data_i = squeeze(data_ic{c}); 
            if isvector(data_i) 
                data_i = data_i(:); 
                notNAN = [notNAN ; numel(data_i)]; 
                if numel(data_i) >= min_datapoints % Shrink to size of min_datapoints 
                    data_i = data_i(ceil(linspace(1,numel(data_i)   ,min(min_datapoints,numel(data_i))))); 
                else % or blow up 
                    data_i = [data_i; nan(min_datapoints-numel(data_i),1)]; 
                end 
                 
                data = [data ; data_i']; 
                header{end+1} = [ var_names{i}, '_c' num2str(c), ' ']; 
                printstr = [printstr '%.10f ']; 
            else if size(data_i,3)==1 
                     if size(data_i,2)>size(data_i,1), data_i =data_i';end 
                    for j=1:size(data_i,2) 
                        notNAN = [notNAN ; numel(data_i(:,j))]; 
                        if size(data_i,1) >= min_datapoints % Shrink to size of min_datapoints 
                            data_ij = data_i(ceil(linspace(1,numel(data_i(:,j)),min(min_datapoints, numel(data_i(:,j))))),j); 
                        else % or blow up 
                            data_ij = [data_i(:,j); nan(min_datapoints-numel(data_i(:,j)),1)]; 
                        end 
                        data = [data ; data_ij']; 
                        header{end+1} = [var_names{i}, '_c' num2str(c), '_' num2str(j), ' ']; 
                        printstr = [printstr '%f ']; 
                    end 
                else 
                    warning('only 2D matrices are currently handled'); 
                end 
            end 
        end 
         
    else 
%         data_i = squeeze(data_i); 
        if size(data_i,2)==1 
            data_i = data_i(:); 
            notNAN = [notNAN ; numel(data_i)]; 
            if numel(data_i) >= min_datapoints % Shrink to size of min_datapoints 
                data_i = data_i(ceil(linspace(1,numel(data_i)   ,min(min_datapoints,numel(data_i))))); 
            else % or blow up 
                data_i = [data_i; nan(min_datapoints-numel(data_i),1)]; 
            end 
             
            data = [data ; data_i']; 
            header{end+1} = [ var_names{i}, ' ']; 
            printstr = [printstr '%.10f ']; 
        else if size(data_i,3)==1 
                % if size(data_i,2)>size(data_i,1), data_i =data_i';end 
                for j=1:size(data_i,2) 
                    notNAN = [notNAN ; numel(data_i(:,j))]; 
                    if size(data_i,1) >= min_datapoints % Shrink to size of min_datapoints 
                        data_ij = data_i(ceil(linspace(1,numel(data_i(:,j)),min(min_datapoints, numel(data_i(:,j))))),j); 
                    else % or blow up 
                        data_ij = [data_i(:,j); nan(min_datapoints-numel(data_i(:,j)),1)]; 
                    end 
                    data = [data ; data_ij']; 
                    header{end+1} = [var_names{i}, '_' num2str(j), ' ']; 
                    printstr = [printstr '%f ']; 
                end 
            else 
                warning('only 2D matrices are currently handled'); 
            end 
        end 
    end 
end 
 
 
printstr = [printstr '\n']; 
 
[~,I]=sort(notNAN,'descend'); 
data = max(min(data(I,:),opt.maxval,'includenan'),opt.minval,'includenan'); 
head = []; 
for i=1:numel(I) 
    head = [head header{I(i)}]; 
end 
head = [head '\n']; 
fileID = fopen([filename '.txt'],'w'); 
fprintf(fileID,head); 
fprintf(fileID,printstr,data); 
fclose(fileID); 
 
fileID =  fopen([filename '.txt'],'r'); 
f=fread(fileID,'*char')'; fclose(fileID); 
f = strrep(f,'NaN',emptyfill); 
fileID =  fopen([filename '.txt'],'w'); 
fprintf(fileID,'%s',f); 
fclose(fileID); 
% end 
