function dataTable = container2table(mapObj, columnNames)
    keysArray = keys(mapObj);
    valuesArray = values(mapObj);
    arrayLengths = cellfun(@length, valuesArray);
    assert(all(arrayLengths == arrayLengths(1)), 'All arrays must have the same length.');

    tableData = cell(length(keysArray), numel(columnNames) + 1);
    for i = 1:length(keysArray)
        tableData{i, 1} = keysArray{i};
        tableData(i, 2:end) = num2cell(valuesArray{i});
    end

    % Correctly formatting the variable names
    variableNames = ['Key', columnNames];
    if iscell(columnNames)
        variableNames = ['Key', columnNames{:}]; % Concatenating 'Key' with the elements of columnNames
    end
    
    dataTable = cell2table(tableData, 'VariableNames', variableNames);
end