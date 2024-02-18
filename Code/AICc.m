function result = AICc(k,n,L2)
    result = AIC(k,n,L2) + ((2 * k * (k+1)) / (n-k-1));
end