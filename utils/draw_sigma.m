    % First create capAt, the lower-triangular matrix A(t) with ones on the
    % main diagonal. This is because the vector Atdraw has a draw of only the
    % non-zero and non-one elements of A(t) only.
    capAt = zeros(M*t,M);
    for i = 1:t
        capatemp = eye(M);
        aatemp = Atdraw(:,i);
        ic=1;
        for j = 2:M
            capatemp(j,1:j-1) = aatemp(ic:ic+j-2,1)';
            ic = ic + j - 1;
        end
        capAt((i-1)*M+1:i*M,:) = capatemp;
    end
    
    % yhat is the vector y(t) - Z x B(t) defined previously. Multiply yhat
    % with capAt, i.e the lower triangular matrix A(t). Then take squares
    % of the resulting quantity (saved in matrix y2)
    y2 = [];
    for i = 1:t
        ytemps = capAt((i-1)*M+1:i*M,:)*yhat(:,i);
        y2 = [y2  (ytemps.^2)]; %#ok<AGROW>
    end   

    % Transform to the log-scale but also add the 'offset constant' to prevent
    % the case where y2 is zero (in this case, its log would be -Infinity) 
    yss = zeros(t,M);
    for i = 1:M
        yss(:,i) = log(y2(i,:)' + 0.001);
    end
    
    % In order to draw the log-volatilies, substract the mean and variance
    % of the 7-component mixture of Normal approximation to the measurement
    % error covariance
    vart = zeros(t*M,M);
    yss1 = zeros(t,M);
    for i = 1:t
        for j = 1:M
            imix = statedraw(i,j);
            vart((i-1)*M+j,j) = u2_s(imix);
            yss1(i,j) = yss(i,j) - m_s(imix) + 1.2704;
        end
    end
    
    % Sigtdraw is a draw of the diagonal log-volatilies, which will give us SIGMA(t)
    [Sigtdraw,log_lik3] = carter_kohn(yss1',Zs,vart,Wdraw,M,M,t,sigma_prmean,sigma_prvar);
    
    % Next draw statedraw (chi square approximation mixture component) conditional on Sigtdraw
    % This is used to update at the next step the log-volatilities Sigtdraw
    for jj = 1:M
        for i = 1:t
            for j = 1:numel(m_s)
                temp1= (1/sqrt(2*pi*u2_s(j)))*exp(-.5*(((yss(i,jj) - Sigtdraw(jj,i) - m_s(j) + 1.2704)^2)/u2_s(j)));
                prw(j,1) = q_s(j,1)*temp1;
            end
            prw = prw./sum(prw);
            cprw = cumsum(prw);
            trand = rand(1,1);
            if trand < cprw(1,1); imix=1;
            elseif trand < cprw(2,1), imix=2;
            elseif trand < cprw(3,1), imix=3;
            elseif trand < cprw(4,1), imix=4;
            elseif trand < cprw(5,1), imix=5;
            elseif trand < cprw(6,1), imix=6;
            else imix=7; 
            end
            statedraw(i,jj)=imix;  % this is a draw of the mixture component index
        end
    end
    
    % Draws in Sigtdraw are in logarithmic scale (log-volatilies). Create 
    % original standard deviations of the VAR covariance matrix
    sigtemp = eye(M);
    sigt = zeros(M*t,M);
    for i = 1:t
        for j = 1:M
            sigtemp(j,j) = exp(0.5*Sigtdraw(j,i));
        end
        sigt((i-1)*M+1:i*M,:) = sigtemp;
    end

    %=====| Draw W, the covariance of SIGMA(t) (from iWishart)
    % Get first differences of Sigtdraw to compute the SSE
    Sigttemp = Sigtdraw(:,2:t)' - Sigtdraw(:,1:t-1)';

    sse_2 = zeros(M,M);
    for i = 1:t-1
        sse_2 = sse_2 + Sigttemp(i,:)'*Sigttemp(i,:);
    end
    Winv = inv(sse_2 + W_prmean);
    Winvdraw = wish(Winv,t+W_prvar);
    Wdraw = inv(Winvdraw);  % this is a draw from W