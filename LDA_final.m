[trainingdata, testdata] = PCA_reduction();

% Make 3d array 1d: each image 2d: each feature 3d: each class
classes = [];
i = 1;
for k = (1:5:size(trainingdata,2))
    classes(:,:,i) = trainingdata(:,k:(k+4))';
    i = i + 1;
end

% Number of observations of each class
num = [];
for i = 1:(size(classes,3))
    num(end+1) = size(classes, 1);
end

%Mean of each class
mu_n = [];
for i = 1:(size(classes,3))
    mu_n(1,:,i) = mean(classes(:,:,i));
end

% Average of the mean of all classes (mean of entire data)
mu_a = mu_n(1,:,1);
for i = 2:(size(classes,3))
    mu_a(1,:) = mu_a(1,:) + mu_n(1,:,i);
end
mu_a = mu_a / (size(classes,3));
    

% Center the data on respective class mean (data-mean)
d_n = [];
for i = 1:(size(classes,3))
    d_n(:,:,i) = classes(:,:,i)-repmat(mu_n(:,:,i),size(classes(:,:,i),1),1);
end

% Calculate the within class scatter for each class
sw_n = [];
for i = 1:(size(classes,3))
    sw_n(:,:,i) = d_n(:,:,i)'*d_n(:,:,i);
end

% Calculate within class scatter for entire data set (SW)
sw_a = sw_n(:,:,1);
for i = 2:(size(classes,3))
    sw_a = sw_a(:,:) + sw_n(:,:,i);
end
invswa = inv(sw_a); % get inverse for use in calculating eigen vector

% Calculate distance from each class mean to all data mean
sb_n = [];
for i = 1:(size(classes,3))
   sb_n(:,:,i) = num(i)*((mu_n(:,:,i)-mu_a(:,:))'*(mu_n(:,:,i)-mu_a(:,:)));
end

% Combine all of these to get between class scatter (SB)
sb_a = sb_n(:,:,1);
for i = 2:(size(classes,3))
    sb_a(:,:) = sb_a(:,:) + sb_n(:,:,i);
end

v = invswa*sb_a;

% find eigen values and eigen vectors of the (v)
[eigvector,eval]=eig(v);
% Sort eigenvalues and vectors descending
eigvalue = diag(eval);
[junk, index] = sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);

% Get eigenvectors with non-zero eigenvalues
count1=0;
for i=1:size(eigvalue,1)
    if(eigvalue(i)>0)
        count1=count1+1;
    end
end

% Calculate proportion of variance from non zero eigenvalues
% and get input from user on how many vectors to use
nonzeig = eigvalue(1:count1);
if (length(nonzeig) > 1)
    results = [];
    tempvecs = [];
    sumnz = sum(nonzeig);
    for i = 1:(length(nonzeig))
        tempvecs(end+1) = nonzeig(i);
        sumt = sum(tempvecs);
        r = sumt/sumnz;
        results(end+1) = r;
    end
    numvec = (1:length(results));
    plot(numvec, results);
    out = 'Enter number of vectors to use: ';
    innum = input(out);
end

vec = eigvector(:,1:innum);
% combine each mean norm. class data into one matrix
tdata = [];
for i=1:size(d_n, 3)
    for r=1:size(d_n, 1)
        tdata(end+1,:) = d_n(r,:,i);
    end
end
% transpose data for projection into fisher space
tdata = tdata';
fdata = vec'*tdata;

% Get mean of transformed data and mean normalize test data with it
% Mean of transformed data
tmean = mean(fdata);
testdata = testdata-repmat(tmean,size(testdata,1),1);
% Project test data into fisher space
ftestdata = vec'*testdata;

%  construct matrix to indicate class of each test image
actuallabels = zeros([40 200]);
for j=1:200
    i = ceil(j/5);
    actuallabels(i,j) = 1;
end

% Calculate distance of test image to each training image
predictlabels = zeros([40 200]);
for i=1:size(ftestdata,2) % Use each test image
    for c=0:39 % calculate min dist for each class
        alldata = ftestdata(:,i)'; % Insert testing image
        cldata = [];
        for ci=1:5 
            cldata(end+1,:) = fdata(:,(c*5)+ci)';
        end
        alldata(2:6,:) = cldata; % Insert each data point for each class 
        dist = pdist(alldata); 
        [a,b] = min(dist(:,1:5)); % Get min dist for each class
        predictlabels(c+1,i) = a; % Insert min dist into prediction matrix
    end
end
% Since roc takes only values in [0, 1] we normalize with largest dist in
% prediction matrix
maxdist = max(max(predictlabels));
predictlabels = predictlabels / maxdist;
% Since roc takes the biggest value to be the predicted class we take the
% complement so the smallest distance is selected instead of largest
onearray = ones(40,200);
predictlabels = onearray - predictlabels;

[tpr, fpr, thresholds] = roc(actuallabels, predictlabels);
plotroc(actuallabels, predictlabels);

% View each class one at a time
% for i=1:40
%     plot(cell2mat(fpr(i)),cell2mat(tpr(i)));
%     input('press enter');
% end
