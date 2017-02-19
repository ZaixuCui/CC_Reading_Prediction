
clear

load GMV_S500_All_training.mat;
load Weight.mat;
Weight_NonZero_Index = find(Weight);
Weight_Zero_Index = setdiff([1:length(Weight)], Weight_NonZero_Index);
Correlated_Voxels_Quantity = 0;
for i = 1:length(Weight_NonZero_Index)
    i
    Correlation = corr(GMV_S500_All_training(:, Weight_NonZero_Index(i)), GMV_S500_All_training(:, Weight_Zero_Index));
    BiggerThan_095 = find(Correlation >= 0.95);
    if  ~isempty(BiggerThan_095)
        for j = 1:length(BiggerThan_095)
            Correlated_Voxels_Quantity = Correlated_Voxels_Quantity + 1;
            Correlated_Voxels_Index(Correlated_Voxels_Quantity) = Weight_Zero_Index(BiggerThan_095(j));
            Original_Voxels_Index(Correlated_Voxels_Quantity) = Weight_NonZero_Index(i);
        end 
    end
end
save Correlated_Voxels_Index_095.mat Correlated_Voxels_Index;
save Original_Voxels_Index_095.mat Original_Voxels_Index;

Correlated_Voxels_Unique.index = unique(Correlated_Voxels_Index);
for i = 1:length(Correlated_Voxels_Unique.index)
    indice = find(Correlated_Voxels_Index == Correlated_Voxels_Unique.index(i));
    Weight_tmp = 0;
    for j = 1:length(indice)
        Weight_tmp = Weight_tmp + Weight(Original_Voxels_Index(indice(j)));
    end
    Weight_tmp = Weight_tmp / length(indice);
    Correlated_Voxels_Unique.Weight(i) = Weight_tmp;
end
save Correlated_Voxels_Unique_095.mat Correlated_Voxels_Unique;

Weight_New = Weight;
Weight_New(Correlated_Voxels_Unique.index) = Correlated_Voxels_Unique.Weight;
save Weight_New_095.mat Weight_New;

