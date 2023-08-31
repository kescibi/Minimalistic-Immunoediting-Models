function size_cells = convert2cells(size)
%CONVERT2CELLS Converts [ml] = [cm^3] -> no. of cells
    size_cells = size * 10^9;
end

