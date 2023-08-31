clear all
close all
%clc

%% load data
load('prediction_table_n_r2_approach_0.mat');
pred_0 = table_n_r2;
load('prediction_table_n_r2_approach_1.mat');
pred_1 = table_n_r2;
load('prediction_table_n_r2_approach_2.mat');
pred_2 = table_n_r2;
load('prediction_table_n_r2_approach_3.mat');
pred_3 = table_n_r2;
load('table_n_r2_approach_0.mat');
prog_0 = table_n_r2;
load('table_n_r2_approach_1.mat');
prog_1 = table_n_r2;
load('table_n_r2_approach_2.mat');
prog_2 = table_n_r2;
load('table_n_r2_approach_3.mat');
prog_3 = table_n_r2;
clear table_n_r2

%% assign groups and perform two sample t tests
% wir wollen Aussage treffen, ob Approach X besser als Approach Y ist.
% Machen also einen Test für alle R^2 der einzelnen approaches (Spalte 3).

% delete values that are -Inf: all values larger cutoff value K are
% considered only
K = 0 % Cutoff von -0.7
%K = -1e40 % betrachtet alles, ausser -Inf

% t test ist falsch hier: Dafuer muessten alle R^2 normalverteilt sein.
% [Th01,Tp01] = ttest2(pred_0(pred_0(:,3)>K,3),pred_1(pred_1(:,3)>K,3))
% [Th02,Tp02] = ttest2(pred_0(pred_0(:,3)>K,3),pred_2(pred_2(:,3)>K,3))
% [Th03,Tp03] = ttest2(pred_0(pred_0(:,3)>K,3),pred_3(pred_3(:,3)>K,3))
% [Th12,Tp12] = ttest2(pred_1(pred_1(:,3)>K,3),pred_2(pred_2(:,3)>K,3))
% [Th13,Tp13] = ttest2(pred_1(pred_1(:,3)>K,3),pred_3(pred_3(:,3)>K,3))
% [Th23,Tp23] = ttest2(pred_2(pred_2(:,3)>K,3),pred_3(pred_3(:,3)>K,3))
% 
% [THp01,TP01] = ttest2(prog_0(prog_0(:,3)>K,3),prog_1(prog_1(:,3)>K,3))
% [TH02,TP02] = ttest2(prog_0(prog_0(:,3)>K,3),prog_2(prog_2(:,3)>K,3))
% [TH03,TP03] = ttest2(prog_0(prog_0(:,3)>K,3),prog_3(prog_3(:,3)>K,3))
% [TH12,TP12] = ttest2(prog_1(prog_1(:,3)>K,3),prog_2(prog_2(:,3)>K,3))
% [TH13,TP13] = ttest2(prog_1(prog_1(:,3)>K,3),prog_3(prog_3(:,3)>K,3))
% [TH23,TP23] = ttest2(prog_2(prog_2(:,3)>K,3),prog_3(prog_3(:,3)>K,3))

% Wilcoxon Rank Sum Test
% Siehe https://de.mathworks.com/help/stats/ranksum.html#bti4z5t
% Bedeutung der errechneten Werte:
%   W steht fuer den Wilcoxon Test
%   p fuer den p Wert
%   h fuer die ausgewaehlte Hypothese
%   Zahl XY bedeutet: Approach X wird mit Y verglichen
% Beispiel: 
%   [Wp03,Wh03] sind der p-Wert und die ausgewaehlte Hypothese fuer den
%   Wilcoxon Test, wenn man approach 0 mit approach 3 zu einem Cutoff Value
%   von K vergleicht (d.h. alle R^2>K werden mit betrachtet).
[Wp01,Wh01] = ranksum(pred_0(pred_0(:,3)>K,3),pred_1(pred_1(:,3)>K,3));
[Wp02,Wh02] = ranksum(pred_0(pred_0(:,3)>K,3),pred_2(pred_2(:,3)>K,3));
[Wp03,Wh03] = ranksum(pred_0(pred_0(:,3)>K,3),pred_3(pred_3(:,3)>K,3));
[Wp12,Wh12] = ranksum(pred_1(pred_1(:,3)>K,3),pred_2(pred_2(:,3)>K,3));
[Wp13,Wh13] = ranksum(pred_1(pred_1(:,3)>K,3),pred_3(pred_3(:,3)>K,3));
[Wp23,Wh23] = ranksum(pred_2(pred_2(:,3)>K,3),pred_3(pred_3(:,3)>K,3));

[WP01,WH01] = ranksum(prog_0(prog_0(:,3)>K,3),prog_1(prog_1(:,3)>K,3));
[WP02,WH02] = ranksum(prog_0(prog_0(:,3)>K,3),prog_2(prog_2(:,3)>K,3));
[WP03,WH03] = ranksum(prog_0(prog_0(:,3)>K,3),prog_3(prog_3(:,3)>K,3));
[WP12,WH12] = ranksum(prog_1(prog_1(:,3)>K,3),prog_2(prog_2(:,3)>K,3));
[WP13,WH13] = ranksum(prog_1(prog_1(:,3)>K,3),prog_3(prog_3(:,3)>K,3));
[WP23,WH23] = ranksum(prog_2(prog_2(:,3)>K,3),prog_3(prog_3(:,3)>K,3));


%median()


%% plot data
% Stimmen alle im jeweiligen Figure ueberein. Scheint also alles zu passen
% :-)

% figure
% subplot(2,2,1)
% histogram(pred_0(:,2))
% hold on
% title('prediction approach 0')
% subplot(2,2,2)
% histogram(pred_1(:,2))
% hold on
% title('prediction approach 1')
% subplot(2,2,3)
% histogram(pred_2(:,2))
% hold on
% title('prediction approach 2')
% subplot(2,2,4)
% histogram(pred_3(:,2))
% hold on
% title('prediction approach 3')
% sgtitle('Number of data points used to fit')
% 
% figure
% subplot(2,2,1)
% histogram(prog_0(:,2))
% hold on
% title('full data approach 0')
% subplot(2,2,2)
% histogram(prog_1(:,2))
% hold on
% title('full data approach 1')
% subplot(2,2,3)
% histogram(prog_2(:,2))
% hold on
% title('full data approach 2')
% subplot(2,2,4)
% histogram(prog_3(:,2))
% hold on
% title('full data approach 3')
% sgtitle('Number of data points used to fit')
