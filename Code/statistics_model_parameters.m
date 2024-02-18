function stats = statistics_model_parameters(preds_all)
    shape = size(preds_all);
    assert(shape(2) == 410);
    stats = zeros(shape(1), 4)
    for i = 1 : shape(1)
        stats(i, :) = [mean(preds_all(i,:)), std(preds_all(i,:)), min((preds_all(i,:))), max((preds_all(i,:)))];
    end
end

