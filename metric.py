from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score, f1_score, confusion_matrix

def metric(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    Acc, P, R, F1, _ = CM_metric(CM)
    
    return Acc, P, R, F1, CM
    
def CM_metric(CM):
    all_ = CM.sum()
    
    Acc = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2] + CM[3][3]) / all_
    P = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0][0:3].sum() + CM[1][0:3].sum() + CM[2][0:3].sum() + CM[3][0:3].sum())
    R = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0].sum() + CM[1].sum() + CM[2].sum())
    F1 = 2 * P * R / (P + R)
    
    return Acc, P, R, F1, CM