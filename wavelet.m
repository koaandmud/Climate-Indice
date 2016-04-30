clear;
load q.txt
sst =q;
n=length(q);
variance = std(sst)^2;
sst = (sst - mean(sst))/sqrt(variance);
dt = 1;
year = [0:n-1]+ 1962.0 ;
xlim = [1962,2011];   %
pad = 1;      
dj = 1/12;     
s0 = 1/2*dt;   
j1 = 6.5/dj;   
lag1 = 0.72;  
mother = 'Morlet';
[wave,period,scale,coi] = wavelet(sst,dt,pad,dj,s0,j1,mother);
power = (abs(wave)).^2 ;  %计算小波系数的模的平方
modulus=abs(wave);  %计算小波系数的模
variance1=sum(power')/n;%计算小波方差 power`共轭转置
%画小波系数实部等值线图
subplot(3,1,1)
levels = [0,0.5,1.0,1.5,2.0,2.5];
v = [0,0.5,1.0,1.5];
Yticks = 0:5:45;
[c,h]=contour(year,period,real(wave),levels,'k-');
clabel(c,h,v,'fontsize',5);
xlabel('年份/year')
ylabel('周期/年 period/year')
title('(a)')
set(gca,'XLim',xlim(:))
set(gca,'YLim',[0 50], ...
    'YDir','default', ...
'YTick',Yticks(:), ...
'YTickLabel',Yticks)
hold on
levels = [-0.5,-1.0,-1.5,-2.0,-2.5];
v = [-0.5,-1.0,-1.5];
[c,h] = contour(year,period,real(wave),levels,'r--');
clabel(c,h,v,'fontsize',5);
hold on
% 画小波方差图
subplot(3,1,2)
plot(period,variance1,'k-')
hold on;
levels= [1,5,10,15,20,25,30,35,40,45];
title('(b)')
set(gca,'XLim',[1,50], ...        
    'XTick',levels,...
   'XTickLabel',levels)
xlabel('周期/a')
ylabel('方差 variance')
hold on
%画小波系数模
subplot(3,1,3)
levels = [0,0.5,1.0,1.5,2.0,2.5];
v = [0,0.5,1.0,1.5];
Yticks = [0:5:30];
[c,h]=contour(year,period,abs(wave),levels,'k-');
clabel(c,h,v,'fontsize',5);
title('(c)')
xlabel('年份/year')
ylabel('周期/年 period/year')
set(gca,'XLim',xlim(:))
set(gca,'YLim',[0 30], ...
    'YDir','default', ...
'YTick',Yticks(:), ...
'YTickLabel',Yticks)
