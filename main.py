import yaml
import os
import json
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Create embedding model using the new HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_real_data_distributions():
    """
    Load and analyze age and sex distributions from real data.
    Returns age distribution parameters and sex proportions.
    """
    real_data_dir = 'real_data'
    files = [f for f in os.listdir(real_data_dir) if f.endswith('.csv')]
    
    all_ages = []
    all_sexes = []
    
    for f in files:
        df = pd.read_csv(os.path.join(real_data_dir, f))
        ages = df['W1_age_r'].dropna()
        sexes = df['W1_sex_r'].dropna()
        
        all_ages.extend(ages.tolist())
        all_sexes.extend(sexes.tolist())
    
    all_ages = np.array(all_ages)
    all_sexes = np.array(all_sexes)
    
    # Calculate age distribution parameters
    age_mean = all_ages.mean()
    age_std = all_ages.std()
    age_min = all_ages.min()
    age_max = all_ages.max()
    
    # Calculate sex proportions
    sex_counts = pd.Series(all_sexes).value_counts(normalize=True).sort_index()
    male_prop = sex_counts.get(1.0, 0)
    female_prop = sex_counts.get(2.0, 0)
    
    return {
        'age_mean': age_mean,
        'age_std': age_std,
        'age_min': age_min,
        'age_max': age_max,
        'male_prop': male_prop,
        'female_prop': female_prop,
        'age_distribution': all_ages  # For empirical sampling
    }

def sample_age_empirical(n_samples, age_distribution):
    """
    Sample ages from the empirical distribution of real data.
    """
    return np.random.choice(age_distribution, size=n_samples, replace=True)

def sample_sex(n_samples, male_prop, female_prop):
    """
    Sample sex from the real data proportions.
    Returns 1 for male, 2 for female.
    """
    return np.random.choice([1, 2], size=n_samples, p=[male_prop, female_prop])

def load_vector_store(mode):
    """Load the appropriate FAISS vector store for the given mode."""
    base_path = "faiss_index"
    if mode == "dsm5+icd10":
        index_path = os.path.join(base_path, "dsm5_icd10_combined")
    elif mode == "dsm5":
        index_path = os.path.join(base_path, "dsm5_only")
    elif mode == "icd10":
        index_path = os.path.join(base_path, "icd10_only")
    elif mode == "none":
        return None
    else:
        raise ValueError(f"Unknown vector store mode: {mode}")
    
    # Use the embedding model directly instead of a function
    faiss_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = faiss_store.as_retriever(similar_text_threshold=0.5)
    return retriever

def run_simulation(module_name, n_patients, vector_store, output_dir, llm_backend, vector_store_mode):
    from agents.patient_agent import PatientAgent

    # Load questionnaire
    with open("documents/questionnaire.json", "r", encoding="utf-8") as f:
        questionnaire = json.load(f)

    # Find the selected module
    if module_name not in questionnaire:
        raise ValueError(f"Module '{module_name}' not found in questionnaire.")
    
    module_data = questionnaire[module_name]
    
    # Load config to get severity distributions
    config = load_config()
    distributions = config.get("distributions", {})
    
    # Get severity distribution for this module, or use default
    if module_name in distributions:
        distribution = distributions[module_name]
        # All distributions now have 5 items, so use 5 severity levels
        severity_levels = ["None", "mild", "moderate", "severe", "very_severe"]
    else:
        # Default distribution if not found in config
        severity_levels = ["None", "mild", "moderate", "severe"]
        distribution = [0.65, 0.05, 0.1, 0.2]
    
    # Normalize distribution to sum to 1
    total = sum(distribution)
    if total > 0:
        distribution = [d/total for d in distribution]
    else:
        distribution = [1.0] + [0.0] * (len(severity_levels) - 1)
    
    # Generate patient demographics
    # ages = np.random.normal(14.86, 1.385, n_patients) # Original line commented out
    # ages = np.clip(ages, 13, 18) # Original line commented out
    # ages = np.round(ages, 0) # Original line commented out
    # sexes = np.random.choice(['male', 'female'], p=[0.314, 0.686], size=n_patients) # Original line commented out

    # Use empirical sampling for age and sex
    real_data_dist = load_real_data_distributions()
    ages = sample_age_empirical(n_patients, real_data_dist['age_distribution'])
    sexes = sample_sex(n_patients, real_data_dist['male_prop'], real_data_dist['female_prop'])

    def run_self_assessment(module_data, severity_level, age, sex, vector_store, llm_backend):
        """Run self-assessment for a patient"""
        patient = PatientAgent(module_data, severity_level, vector_store, age, sex, llm_backend,config)
        
        results = {
            "module": module_name,
            "items": [],
            "responses": {},
            "scores": {}
        }
        
        # Process each item in the questionnaire
        for item in module_data["items"]:
            question = item["label"]
            item_key = item["key"]
            
            # Get patient response (now returns dict with text and score)
            patient_response = patient.respond(question)
            
            # Store the response
            results["responses"][item_key] = patient_response["text"]
            results["scores"][item_key] = patient_response["score"]
            results["items"].append({
                "key": item_key,
                "question": question,
                "response_text": patient_response["text"],
                "response_score": patient_response["score"]
            })
        
        return results

    # Initialize results storage
    all_results = []
    
    for p_id in tqdm(range(n_patients), desc=f"Simulating patients for {module_name}"):
        age = int(ages[p_id])
        sex_num = int(sexes[p_id])  # Convert to Python int
        # Convert numeric sex to string for PatientAgent
        sex_str = "male" if sex_num == 1 else "female"
        severity = np.random.choice(severity_levels, p=distribution)
        
        # Run self-assessment
        assessment_results = run_self_assessment(
            module_data, severity, age, sex_str, vector_store, llm_backend
        )
        
        # Add patient metadata (use numeric sex for CSV output to match real data)
        assessment_results["patient_id"] = p_id + 1
        assessment_results["age"] = age
        assessment_results["sex"] = sex_num  # Keep numeric for CSV
        assessment_results["severity"] = severity
        
        all_results.append(assessment_results)
        
        # Print progress
        print(f"Patient {p_id+1} - {module_name} - Severity: {severity}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as JSON with vector store mode in filename
    json_output_path = os.path.join(output_dir, f"{module_name.lower().replace(' ', '_')}_{vector_store_mode}_dataset.json")
    with open(json_output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Convert to CSV format for compatibility with real data
    csv_data = []
    for result in all_results:
        row = {
            "Patient": result["patient_id"],
            "age": result["age"],
            "sex": result["sex"],
            "severity": result["severity"],
            "module": result["module"]
        }
        
        # Add only scores (matching real data format)
        for item_key, score in result["scores"].items():
            row[item_key] = score
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_output_path = os.path.join(output_dir, f"w1_{module_name.lower().replace(' ', '_')}_{vector_store_mode}_dataset.csv")
    df.to_csv(csv_output_path, index=False)
    
    print(f"Results saved to {json_output_path} and {csv_output_path}")
    return df

def calculate_average_scores(df):
    """
    Calculate average scores for each item in the dataset.
    """
    # Get all columns that are item scores (excluding metadata columns)
    score_columns = [col for col in df.columns if col.startswith('W1_') or col.startswith('w1_')]
    
    if not score_columns:
        # If no W1_ columns found, look for other score columns
        score_columns = [col for col in df.columns if 'it' in col.lower() and col not in ['age', 'sex', 'severity', 'module', 'Patient']]
    
    if not score_columns:
        print("No score columns found!")
        return {}
    
    averages = {}
    for col in score_columns:
        if col in df.columns:
            avg_score = df[col].mean()
            averages[col] = round(avg_score, 3)
    
    return averages

def test_with_10_patients_dsm5():
    """
    Test the system with 10 patients using dsm5 vector store mode.
    """
    print("=== Testing with 10 patients using DSM-5 vector store ===")
    
    # Load config
    config = load_config()
    
    # Override settings for this test
    test_config = config.copy()
    test_config["n_patients"] = 10
    test_config["vector_store_mode"] = ["dsm5"]
    test_config["single_disorder"] = "DEPRESSION"
    
    print(f"Test configuration:")
    print(f"- Number of patients: {test_config['n_patients']}")
    print(f"- Vector store mode: {test_config['vector_store_mode']}")
    print(f"- Disorder: {test_config['single_disorder']}")
    print(f"- LLM backend: {test_config.get('llm_backend', 'ollama')}")
    
    # Load real data distributions for age and sex sampling
    distributions = load_real_data_distributions()
    
    # Sample ages and sexes for 10 patients
    ages = sample_age_empirical(10, distributions['age_distribution'])
    sexes = sample_sex(10, distributions['male_prop'], distributions['female_prop'])
    
    print(f"\nSampled demographics:")
    print(f"- Ages: {ages.tolist()}")
    print(f"- Sexes: {sexes.tolist()} (1=male, 2=female)")
    
    # Load vector store
    vector_store = load_vector_store("dsm5")
    
    # Run simulation for DEPRESSION
    module_name = "DEPRESSION"
    module_names = [module_name]
    
    all_results = []
    
    for module_name in module_names:
        print(f"\nProcessing {module_name} with dsm5 vector store...")
        
        # Load questionnaire
        with open("documents/questionnaire.json", "r", encoding="utf-8") as f:
            questionnaire = json.load(f)
        
        # Find the selected module
        if module_name not in questionnaire:
            print(f"Module {module_name} not found in questionnaire!")
            continue
        
        module_data = questionnaire[module_name]
        severity_levels = ["none", "mild", "moderate", "severe", "very_severe"]
        distribution = config["distributions"][module_name]
        
        # Initialize results storage
        for p_id in range(10):
            age = int(ages[p_id])
            sex_num = int(sexes[p_id])
            sex_str = "male" if sex_num == 1 else "female"
            severity = np.random.choice(severity_levels, p=distribution)
            
            print(f"\nPatient {p_id+1}: {age}-year-old {sex_str} with {severity} {module_name}")
            
            # Create patient agent
            from agents.patient_agent import PatientAgent
            patient = PatientAgent(
                module_data=module_data,
                severity=severity,
                retriever=vector_store,
                age=age,
                sex=sex_str,
                llm_backend=test_config.get("llm_backend", "ollama")
            )
            
            # Get individual characteristics
            characteristics = patient.get_individual_characteristics_summary()
            print(f"  Individual characteristics: {characteristics}")
            
            # Run self-assessment
            results = {
                "patient_id": p_id + 1,
                "age": age,
                "sex": sex_num,
                "severity": severity,
                "module": module_name,
                "individual_characteristics": characteristics,
                "responses": {},
                "scores": {},
                "items": []
            }
            
            # Get questions from module data
            items = module_data.get("items", [])
            for item in items:
                item_key = item["key"]
                question = f"Over the past 2 weeks, how often have you {item['label'].lower()}?"
                
                # Get patient response
                patient_response = patient.respond(question)
                
                # Store the response
                results["responses"][item_key] = patient_response["text"]
                results["scores"][item_key] = patient_response["score"]
                results["items"].append({
                    "key": item_key,
                    "question": question,
                    "response_text": patient_response["text"],
                    "response_score": patient_response["score"]
                })
                
                print(f"    {item_key}: Score {patient_response['score']} - {patient_response['text'][:50]}...")
            
            all_results.append(results)
    
    # Convert to DataFrame for analysis
    csv_data = []
    for result in all_results:
        row = {
            "Patient": result["patient_id"],
            "age": result["age"],
            "sex": result["sex"],
            "severity": result["severity"],
            "module": result["module"]
        }
        
        # Add scores
        for item_key, score in result["scores"].items():
            row[item_key] = score
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    
    # Calculate and display average scores
    print(f"\n=== AVERAGE SCORES ===")
    averages = calculate_average_scores(df)
    
    if averages:
        print("Average scores by item:")
        for item, avg_score in averages.items():
            print(f"  {item}: {avg_score}")
        
        # Calculate overall average
        overall_avg = sum(averages.values()) / len(averages)
        print(f"\nOverall average score: {overall_avg:.3f}")
        
        # Calculate average by severity
        print(f"\nAverage scores by severity:")
        for severity in ["none", "mild", "moderate", "severe", "very_severe"]:
            severity_df = df[df['severity'] == severity]
            if not severity_df.empty:
                severity_averages = calculate_average_scores(severity_df)
                if severity_averages:
                    severity_avg = sum(severity_averages.values()) / len(severity_averages)
                    print(f"  {severity.capitalize()}: {severity_avg:.3f}")
    else:
        print("No scores found to calculate averages!")
    
    # Save results
    output_dir = test_config.get("output_dir", "./results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_output_path = os.path.join(output_dir, "test_10_patients_dsm5_results.json")
    with open(json_output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save as CSV
    csv_output_path = os.path.join(output_dir, "test_10_patients_dsm5_results.csv")
    df.to_csv(csv_output_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_output_path}")
    print(f"  CSV: {csv_output_path}")
    
    return df, averages

def main():
    config = load_config()
    print(f"Starting self-assessment simulation with vector store mode: {config['vector_store_mode']}")
    
    # Check if single disorder mode is enabled
    if "single_disorder" in config and config["single_disorder"]:
        module_names = [config["single_disorder"]]
        print(f"Running in single disorder mode: {config['single_disorder']}")
    else:
        module_names = config["module_name"]
        print(f"Running all disorders: {module_names}")
    
    for module_name in module_names:
        for vector_store_mode in config["vector_store_mode"]:
            print(f"\nProcessing {module_name} with {vector_store_mode} vector store...")
            vector_store = load_vector_store(vector_store_mode)
            run_simulation(
                module_name,
                config["n_patients"],
                vector_store,
                config.get("output_dir", "./results"),
                config.get("llm_backend", "ollama"),
                vector_store_mode
            )

if __name__ == "__main__":
    import sys
    
    # Check if test command is provided
    if len(sys.argv) > 1 and sys.argv[1] == "test_10_dsm5":
        test_with_10_patients_dsm5()
    else:
        main() 