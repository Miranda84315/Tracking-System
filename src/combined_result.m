function  combined_result

result = [easy; hard];
result(:, [2, 3]) = result(:, [3, 2]);
result = result(:, 1:9);

root = 'D:/Code/TrackingSystem/experiments/experiment_v7/duke.txt'
dlmwrite(root , result, 'delimiter', ' ', 'precision', '%.2f');

fid=fopen(root, 'w');%建立文件
%循環寫入數據
for i=1:length(result)
	fprintf(fid, '%d %d %d %.0f %.0f %.0f %.0f %.4f %.4f ', result(i, :));
end
fclose(fid); 