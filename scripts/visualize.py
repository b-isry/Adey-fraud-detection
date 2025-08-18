import shap
import matplotlib.pyplot as plt

def generate_summary_plot(model, features, feature_names, plot_title="SHAP Summary Plot", check_additivity=False):
    """
    Generates and displays a SHAP summary plot.
    """
    print(f"Calculating SHAP values for {len(features)} samples. This may take a moment...")
    
    # CORRECTED: Instantiate the explainer without the argument.
    explainer = shap.TreeExplainer(model)
    
    # CORRECTED: Pass check_additivity to the shap_values calculation.
    shap_values = explainer.shap_values(features, check_additivity=check_additivity)

    print("SHAP values calculated. Generating plot...")
    # The explainer for Random Forest returns a list of two arrays (one for each class).
    # We are interested in the SHAP values for the "fraud" class (class 1).
    shap.summary_plot(shap_values[1], features, feature_names=feature_names, show=False)
    plt.title(plot_title)
    plt.show()

    return shap_values

# Keep the generate_force_plot function as is, or update it to be safe
def generate_force_plot(model, features, instance_to_explain):
    """
    Generates and displays a SHAP force plot for a single prediction.
    `instance_to_explain` should be a single-row DataFrame.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance_to_explain)

    # Initialize javascript for plotting in notebooks
    shap.initjs()

    # Create force plot for the specific instance
    return shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][0, :],
        instance_to_explain.iloc[0],
        matplotlib=True
    )