import pandas as pd
import os
import numpy as np
from collections import defaultdict
import json

def load_mimic_data(folder_path):
    """Load relevant MIMIC-III tables with error handling"""
    try:
        admissions = pd.read_csv(os.path.join(folder_path, 'ADMISSIONS.csv'), 
                               parse_dates=['admittime', 'dischtime', 'deathtime'])
        diagnoses = pd.read_csv(os.path.join(folder_path, 'DIAGNOSES_ICD.csv'))
        labevents = pd.read_csv(os.path.join(folder_path, 'LABEVENTS.csv'),
                              parse_dates=['charttime'])
        prescriptions = pd.read_csv(os.path.join(folder_path, 'PRESCRIPTIONS.csv'),
                                  parse_dates=['startdate', 'enddate'])
        return admissions, diagnoses, labevents, prescriptions
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None

def create_severity_mapping():
    """Create a comprehensive ICD9 severity mapping with clinical categories"""
    severity_map = {
        # Infectious diseases
        '038.*': 4, '995.91': 4, '995.92': 4,  # Sepsis
        '487.0': 3, '487.1': 3, '487.8': 3,   # Influenza
        'V09.*': 2,                            # Infection with drug-resistant microorganisms
        
        # Cardiovascular
        '410.*': 4,                            # Acute MI
        '428.*': 4,                            # Heart failure
        '426.*': 3,                            # Conduction disorders
        '414.*': 3,                           # Chronic ischemic heart disease
        
        # Respiratory
        '518.81': 4, '518.82': 4, '518.84': 4, # Acute respiratory failure
        '486': 3,                              # Pneumonia
        '493.*': 2,                            # Asthma
        
        # Renal
        '584.*': 4,                            # Acute kidney failure
        '585.*': 3,                            # Chronic kidney disease
        '586': 2,                              # Renal failure unspecified
        
        # Neurological
        '430': 4, '431': 4,                    # Stroke
        '348.1': 4,                            # Anoxic brain injury
        '780.09': 2,                           # Altered mental status
        
        # Gastrointestinal
        '570': 4,                              # Acute liver failure
        '571.*': 3,                            # Chronic liver disease
        '578.*': 3,                            # GI bleeding
        
        # Default
        'default': 2
    }
    
    # Expand ranges (simplified - in practice would use regex)
    expanded_map = {}
    for code, severity in severity_map.items():
        if '.*' in code:
            prefix = code.split('.')[0]
            expanded_map.update({f"{prefix}.{i}": severity for i in range(10)})
        else:
            expanded_map[code] = severity
    
    return expanded_map

def calculate_lab_severity(lab_data):
    """Calculate lab abnormality severity score"""
    if lab_data.empty:
        return 0
    
    # Define critical lab thresholds
    critical_labs = {
        '50868': (3.5, 5.1),   # Potassium (mEq/L)
        '50912': (70, 110),     # Glucose (mg/dL)
        '50931': (135, 145),    # Sodium (mEq/L)
        '51222': (12, 16),      # Hemoglobin (g/dL)
        '50960': (0.6, 1.2),    # Creatinine (mg/dL)
    }
    
    severity = 0
    for _, lab in lab_data.iterrows():
        itemid = str(lab['itemid'])
        value = lab['valuenum']
        
        if itemid in critical_labs and not np.isnan(value):
            low, high = critical_labs[itemid]
            if value < low * 0.8 or value > high * 1.2:
                severity += 2  # Critical abnormality
            elif value < low * 0.9 or value > high * 1.1:
                severity += 1   # Moderate abnormality
    
    return min(severity, 5)  # Cap at 5

def calculate_treatment_intensity(prescription_data):
    """Calculate treatment intensity score"""
    if prescription_data.empty:
        return 0
    
    # Classify medications by intensity
    high_intensity = ['HEPARIN', 'INSULIN', 'VANCOMYCIN', 'MEROPENEM']
    medium_intensity = ['CEPHALOSPORIN', 'QUINOLONE', 'WARFARIN']
    
    intensity = 0
    for _, med in prescription_data.iterrows():
        drug_name = str(med['drug']).upper()
        
        if any(h in drug_name for h in high_intensity):
            intensity += 3
        elif any(m in drug_name for m in medium_intensity):
            intensity += 2
        else:
            intensity += 1
    
    return min(intensity // 3, 4)  # Normalize to 0-4 scale

def preprocess_data(admissions, diagnoses, labevents, prescriptions, output_folder):
    """Preprocess data and save hospital-specific files"""
    severity_map = create_severity_mapping()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    hospital_data = defaultdict(list)
    
    # Process each admission
    for _, adm in admissions.iterrows():
        hadm_id = adm['hadm_id']
        subject_id = adm['subject_id']
        
        # Get relevant data for this admission
        adm_diagnoses = diagnoses[diagnoses['hadm_id'] == hadm_id]
        adm_labs = labevents[(labevents['hadm_id'] == hadm_id) | 
                           ((labevents['subject_id'] == subject_id) & 
                            (labevents['charttime'] >= adm['admittime']) & 
                            (labevents['charttime'] <= adm['dischtime']))]
        adm_prescriptions = prescriptions[prescriptions['hadm_id'] == hadm_id]
        
        # Calculate diagnosis severity (max severity code)
        diag_severity = max(
            [severity_map.get(code.split('.')[0], severity_map['default']) 
             for code in adm_diagnoses['icd9_code'].astype(str)],
            default=severity_map['default']
        )
        
        # Calculate lab severity
        lab_severity = calculate_lab_severity(adm_labs)
        
        # Calculate treatment intensity
        treatment_intensity = calculate_treatment_intensity(adm_prescriptions)
        
        # Calculate outcome (1 if died, 0 otherwise)
        outcome = 1 if pd.notna(adm['deathtime']) else 0
        
        # Create patient state
        patient_state = {
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'admission_type': adm['admission_type'],
            'diagnosis_severity': diag_severity,
            'lab_severity': lab_severity,
            'treatment_intensity': treatment_intensity,
            'los_hours': (adm['dischtime'] - adm['admittime']).total_seconds() / 3600,
            'outcome': outcome,
            'primary_diagnosis': adm['diagnosis'],
            'discharge_location': adm['discharge_location']
        }
        
        # Group by hospital (using admission location as proxy)
        hospital_id = hash(adm['admission_location']) % 10  # Create 10 hospitals
        hospital_data[hospital_id].append(patient_state)
    
    # Save hospital data to separate files
    for hospital_id, patients in hospital_data.items():
        df = pd.DataFrame(patients)
        df.to_csv(os.path.join(output_folder, f'hospital_{hospital_id}.csv'), index=False)
    
    # Save metadata
    metadata = {
        'severity_map': severity_map,
        'hospital_count': len(hospital_data),
        'total_patients': sum(len(p) for p in hospital_data.values())
    }
    with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    return hospital_data

def process_mimic_data(input_folder, output_folder='hospital_data'):
    """Main function to process MIMIC data"""
    print(f"Processing MIMIC data from {input_folder}")
    
    # Load data
    admissions, diagnoses, labevents, prescriptions = load_mimic_data(input_folder)
    if admissions is None:
        return None
    
    # Preprocess and save hospital data
    hospital_data = preprocess_data(
        admissions, diagnoses, labevents, prescriptions, output_folder)
    
    print(f"\nProcessing complete. Saved data for {len(hospital_data)} hospitals")
    print(f"Total patients processed: {sum(len(p) for p in hospital_data.values())}")
    print(f"Output saved to: {os.path.abspath(output_folder)}")
    
    return hospital_data

if __name__ == "__main__":
    # Configure paths
    input_folder = os.path.join('mimic-iii-clinical-database-demo-1.4', 
                              'mimic-iii-clinical-database-demo-1.4')
    output_folder = 'hospital_data'
    
    # Process data
    process_mimic_data(input_folder, output_folder)