function LD = LD_calc(volume)
    LD = 2*nthroot((3/(4*pi))*1000*convert2ml(volume*10^9),3);
    %LD = 2*nthroot((3/(4*pi))*1000*convert2ml(volume*10^9),3);
end