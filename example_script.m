%% First define parameters to be used in all models:


clear;close all;clc;

params = struct();
params.S=6; %Number of impulse responses to store (number of months)
params.reps=50000; %Number of iterations
params.burns=params.reps/10; %Number of burn ins
params.lags=2; %Number of lags in VAR
params.tau=70; %Size of training sample
params.kq=0.01; %Prior variance on coefficients, refer to Primiceri (2005) page 843
params.ks=0.1; %Prior variance on covariance matrix elements, refer to Primiceri (2005) page 843
params.kw=0.01;%Prior variance on standard deviation of orthogonal innovations, refer to Primiceri (2005) page 843
params.olsprior=1; %1 to use OLS priors, 0 to use non-informative priors
params.stdize=1; %1 to standardize data before estimation, 0 to use data as is
params.trep=12; %Frequency of saving impulse responses. 1 saves for every month, 2 saves for every second month, and so on. Higher frequency uses much more memory.
params.istore=1; %Set to zero to not save any impulse responses at all.
params.nhor= params.S;


save('data/tvp_specs')
disp('Specs have been saved')


%%
% First estimate model with cumulative share flows:

clear;clc;close all;
load('data/flow_data'); %Load data
load('data/tvp_specs'); %Specs

addpath utils

Y_in=[yraw(:,1) cumsum(yraw(:,2)) yraw(:,5)]
names={'VIX';'ShareFlowC';'Spread'}

[N M]=size(Y_in);
figure %Plot input data
for i=1:M
    subplot(3,1,i)
    plot(Y_in(:,i))
    legend(names(i,:))
end

ModelOut= heteroskedastic_time_varying_VAR(Y_in,...
    yearlab,...
    params);

save(['data/Share_Cumulative_Output'])
   
%%

% First estimate model with cumulative bond flows:
% 
% clear;clc;close all;
% load('KOOP_DATA'); %Load data
% load('tvp_specs_priors2')
% y=[yraw(:,1) cumsum(yraw(:,3)) yraw(:,5)]
% 
% names={'VIX';'BondFlowC';'Spread'}
% [N M]=size(y);
% figure %Plot input data
% for i=1:M
%     subplot(3,1,i)
%     plot(y(:,i))
%     legend(names(i,:))
% end
% 
% ModelOut=HTVP_VAR_FUNC(y,yearlab,nhor,reps,burns,lags,tau,ks,kw,kq,olsprior,stdize,trep,istore);
% save(['Data_Output/Bond_Cumulative_Output'])
% 


%% Estimate model with net share flows:

clear;clc;close all;
load('KOOP_DATA'); %Load data
load('tvp_specs')
y=[yraw(:,1) yraw(:,2) yraw(:,5)]

names={'VIX';'NetShareFlow';'Spread'}
[N M]=size(y);
figure %Plot input data
for i=1:M
    subplot(3,1,i)
    plot(y(:,i))
    legend(names(i,:))
end

ModelOut=HTVP_VAR_FUNC(y,yearlab,nhor,reps,burns,lags,tau,ks,kw,kq,olsprior,stdize,trep,istore);
save(['Data_Output/Share_Net_Output'])
   
%%

% First estimate model with net bond flows:
% 
% clear;clc;close all;
% load('KOOP_DATA'); %Load data
% load('tvp_specs')
% y=[yraw(:,1) yraw(:,3) yraw(:,5)]
% 
% names={'VIX';'NetBondFlow';'Spread'}
% [N M]=size(y);
% figure %Plot input data
% for i=1:M
%     subplot(3,1,i)
%     plot(y(:,i))
%     legend(names(i,:))
% end
% 
% ModelOut=HTVP_VAR_FUNC(y,yearlab,nhor,reps,burns,lags,tau,ks,kw,kq,olsprior,stdize,trep,istore);
% save(['Data_Output/Bond_Net_Output'])

%%

%%

% % Reorder cumulative bonds:
% 
% clear;clc;close all;
% load('KOOP_DATA'); %Load data
% load('tvp_specs')
% 
% y=[yraw(:,1)  yraw(:,5) cumsum(yraw(:,3))]
% 
% names={'VIX';'Spread';'BondFlowC'}
% [N M]=size(y);
% figure %Plot input data
% for i=1:M
%     subplot(3,1,i)
%     plot(y(:,i))
%     legend(names(i,:))
% end
% 
% ModelOut=HTVP_VAR_FUNC(y,yearlab,nhor,reps,burns,lags,tau,ks,kw,kq,olsprior,stdize,trep,istore);
% 
% 
% save(['Data_Output/Bond_Cumulative_Output_Order2'])

%%

% Reorder cumulative shares:

% clear;clc;close all;
% load('KOOP_DATA'); %Load data
% load('tvp_specs')
% 
% y=[yraw(:,1)  yraw(:,5) cumsum(yraw(:,2))]
% 
% names={'VIX';'Spread';'ShareFlowC'}
% [N M]=size(y);
% figure %Plot input data
% for i=1:M
%     subplot(3,1,i)
%     plot(y(:,i))
%     legend(names(i,:))
% end
% 
% ModelOut=HTVP_VAR_FUNC(y,yearlab,nhor,reps,burns,lags,tau,ks,kw,kq,olsprior,stdize,trep,istore);
% save(['Data_Output/Share_Cumulative_Output_Order2'])

%%

% % Reorder net bond flows:
% 
% clear;clc;close all;
% load('KOOP_DATA'); %Load data
% load('tvp_specs')
% y=[yraw(:,1) yraw(:,5) yraw(:,3)]
% reps=10000; %Number of iterations
% burns=reps/10; %Number of burn ins
% 
% names={'VIX';'Spread';'NetBondFlow'}
% [N M]=size(y);
% figure %Plot input data
% for i=1:M
%     subplot(3,1,i)
%     plot(y(:,i))
%     legend(names(i,:))
% end
% 
% ModelOut=HTVP_VAR_FUNC(y,yearlab,nhor,reps,burns,lags,tau,ks,kw,kq,olsprior,stdize,trep,istore);
% save(['Data_Output/Bond_Net_Output_Order2'])



%% Reorder net share flows:

% clear;clc;close all;
% load('KOOP_DATA'); %Load data
% load('tvp_specs')
% y=[yraw(:,1) yraw(:,5) yraw(:,2)]
% reps=10000; %Number of iterations
% burns=reps/10; %Number of burn ins
% 
% names={'VIX';'Spread';'NetShareFlow'}
% [N M]=size(y);
% figure %Plot input data
% for i=1:M
%     subplot(3,1,i)
%     plot(y(:,i))
%     legend(names(i,:))
% end
% 
% ModelOut=HTVP_VAR_FUNC(y,yearlab,nhor,reps,burns,lags,tau,ks,kw,kq,olsprior,stdize,trep,istore);
% save(['Data_Output/Share_Net_Output_Order2'])

%%



% Estimate long impulse responses at few intervals:

clear;clc;close all;
load('KOOP_DATA'); %Load data
load('tvp_specs_s24')
y=[yraw(:,1) cumsum(yraw(:,2)) yraw(:,5)]
names={'VIX';'ShareFlowC';'Spread'}
[N M]=size(y);
figure %Plot input data
for i=1:M
    subplot(3,1,i)
    plot(y(:,i))
    legend(names(i,:))
end

ModelOut=HTVP_VAR_FUNC(y,yearlab,nhor,reps,burns,lags,tau,ks,kw,kq,olsprior,stdize,trep,istore);
save(['Data_Output/Share_Cumulative_Output_S24_2'])
   
%%

% First estimate model with cumulative bond flows:

clear;clc;close all;
load('KOOP_DATA'); %Load data
load('tvp_specs_s24')
y=[yraw(:,1) cumsum(yraw(:,3))  yraw(:,5) ]
names={'VIX';'BondFlowC';'Spread'}
[N M]=size(y);
figure %Plot input data
for i=1:M
    subplot(3,1,i)
    plot(y(:,i))
    legend(names(i,:))
end

ModelOut=HTVP_VAR_FUNC(y,yearlab,nhor,reps,burns,lags,tau,ks,kw,kq,olsprior,stdize,trep,istore);
save(['Data_Output/Bond_Cumulative_Output_S24_2'])



%% Estimate long impulse responses at few intervals:

clear;clc;close all;
load('KOOP_DATA'); %Load data
load('tvp_specs_s24')
y=[yraw(:,1) yraw(:,2) yraw(:,5)]
names={'VIX';'NetShareFlow';'Spread'}
[N M]=size(y);
figure %Plot input data
for i=1:M
    subplot(3,1,i)
    plot(y(:,i))
    legend(names(i,:))
end

ModelOut=HTVP_VAR_FUNC(y,yearlab,nhor,reps,burns,lags,tau,ks,kw,kq,olsprior,stdize,trep,istore);
save(['Data_Output/Share_Net_Output_S24_2'])
   
%%

% First estimate model with cumulative bond flows:

clear;clc;close all;
load('KOOP_DATA'); %Load data
load('tvp_specs_s24')
y=[yraw(:,1) yraw(:,3)  yraw(:,5) ]
names={'VIX';'NetBondFlow';'Spread'}
[N M]=size(y);
figure %Plot input data
for i=1:M
    subplot(3,1,i)
    plot(y(:,i))
    legend(names(i,:))
end

ModelOut=HTVP_VAR_FUNC(y,yearlab,nhor,reps,burns,lags,tau,ks,kw,kq,olsprior,stdize,trep,istore);
save(['Data_Output/Bond_Net_Output_S24_2'])

