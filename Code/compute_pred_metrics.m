function metrics = compute_pred_metrics(GT,toEval)
    metrics = zeros(1,3) - 1;
    C = confusionmat(GT, toEval);
    TN = C(1,1); %True Negative -> true no progression
    TP = C(2,2); %True Positive -> True Progressive
    FN = C(2,1); %False Negative -> Predicted no progression, but is progressive
    FP = C(1,2); %False Positive
    
    sensitivity = TP / (TP + FN); %Sensitivity / Recall
    specificity = TN / (TN + FP); %Specificity
    accuracy = (TP + TN) / (TP+TN+FP+FN);

    metrics(1) = sensitivity;
    metrics(2) = specificity;
    metrics(3) = (sensitivity + specificity) / 2;
    metrics(4) = accuracy;
    metrics = round(metrics*100,1);
end

