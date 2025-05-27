import random
from faker import Faker
import pandas as pd
import numpy as np


fake = Faker()

"""
PROMPT 1: 
Here is a mammography report:
Screening Mammogram
Name: Exam Date: Comparison Date and facility: (Example 1/1/12 from Bailey Medical Center) Bilateral Screening mammogram Views: Bilateral CC and MLO (Sometimes will add anterior compression or XCCL/ XCCM views) Breast Composition: (choose 1 of the following statements based on appearance) -Almost entirely fat -Scattered fibroglandular densities -Heterogeneously dense which may obscure detection of small masses -Extremely dense which may lower the sensitivity of mammography Findings: Right breast has no evidence of mass, architectural distortion, suspicious micro calcification, skin thickening or nipple retraction. Left breast has no evidence of mass, architectural distortion, suspicious micro calcification, skin thickening or nipple retraction. Impression: (choose 1 of the following classifications) BIRAD 0 incomplete, need additional imaging evaluation, recommend ________. BIRAD 1 negative, recommend annual screening mammograms. BIRAD 2 benign findings, recommend annual screening mammograms. BIRAD 3 probably benign findings, short interval follow up is suggested, Recommend ________. (6 month f/u diagnostic/ US/ etc.) BIRAD 4 suspicious abnormality, biopsy should be considered. Recommend ________. (Stereotactic/ ultrasound guided/ etc.) BIRAD 5 Highly suggestive of malignancy. Appropriate action should be taken. Recommend ________. (Surgical consultation) BIRAD 6 Known and biopsy proven malignancy.

Here is what I am trying to do:

Create a list of templates for different paraphrased mammography reports
Each of these reports must focus on including the following 6 entities: 1. Name 2. Date 3. Location 4. Hospital 5. Identifier (can be: Medical record number, patient number, etc) 6. Contact
Objective: Create synthetic medical reports in a way that I am able to extract synthetic entities (6 entities) from it to train a language model to be a named entity recognition model
Your goal: synthesize the data like an expert data synthesizer

"""
def generate_mammography_report():
    hospitals = [
        ("Downtown Breast Imaging Center", "123 Main St, Cityville, State"),
        ("Westside Women's Imaging Clinic", "456 Oak Ave, Townsville, State"),
        ("Eastside Diagnostic Center", "789 Pine Rd, Villagetown, State"),
        ("Central Breast Health Facility", "101 Maple Blvd, Metrocity, State"),
        ("Regional Women's Health Center", "555 Cedar Ln, Riverside, State"),
        ("Metropolitan Breast Screening Unit", "777 Birch St, Centraltown, State"),
        ("Community Mammography Services", "999 Elm Dr, Southside, State"),
        ("Advanced Breast Imaging Institute", "321 Willow Ave, Northville, State"),
    ]

    patient_locations = [
        "Cityville", "Townsville", "Villagetown", "Metrocity",
        "Lakeside", "Hilltown", "Riverside", "Centraltown",
        "Southside", "Northville", "Westbrook", "Eastwood"
    ]

    hospital_name, hospital_location = random.choice(hospitals)
    patient_location = random.choice(patient_locations)
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=40, maximum_age=75).strftime("%B %d, %Y")
    address = fake.address().replace("\n", ", ")
    doctor = fake.name()
    phone = fake.phone_number()
    email = fake.email()
    mrn = fake.bothify(text="#####")
    ssn = fake.ssn()
    health_plan_id = fake.bothify(text="########")
    account_number = fake.bothify(text="#######")
    license_number = fake.bothify(text="#######")
    certificate_number = fake.bothify(text="#####")
    license_plate = fake.bothify(text="???###")
    vehicle_id = fake.bothify(text="########")
    device_serial = fake.bothify(text="DEV-SN-######")
    patient_id = fake.bothify(text="PT-####")
    case_number = fake.bothify(text="CN-######")
    study_id = fake.bothify(text="STU-#####")
    scan_date = fake.date()
    scan_type = random.choice(["screening mammogram", "diagnostic mammogram", "bilateral mammography", "unilateral mammography"])
    laterality = random.choice(["left", "right", "bilateral"])
    
    findings = random.choice([
        "no suspicious findings",
        "a focal asymmetry in the upper outer quadrant",
        "scattered fibroglandular densities",
        "a well-circumscribed mass likely benign",
        "an area of architectural distortion requiring further evaluation",
        "benign calcifications in the lower inner quadrant",
        "stable fibroadenoma in the right breast",
        "no evidence of malignancy",
        "heterogeneously dense breast tissue",
        "unremarkable bilateral breast parenchyma"
    ])

    birads = random.choice([
        "BI-RADS 1 (Negative)", 
        "BI-RADS 2 (Benign)", 
        "BI-RADS 3 (Probably benign)",
        "BI-RADS 4 (Suspicious abnormality)",
        "BI-RADS 5 (Highly suggestive of malignancy)"
    ])
    
    follow_up = random.choice([
        "Annual screening recommended.",
        "Ultrasound follow-up in 6 months.",
        "Biopsy recommended for further evaluation.",
        "Short interval follow-up advised.",
        "Continue routine screening.",
        "Diagnostic mammography in 12 months.",
        "Immediate surgical consultation required.",
        "MRI correlation suggested."
    ])

    templates = [
        # Template 1: Standard clinical format
        f"""
MAMMOGRAPHY REPORT
Patient: {name}
DOB: {dob}
MRN: {mrn}
Study Date: {scan_date}
Facility: {hospital_name}, {hospital_location}
Interpreting Physician: Dr. {doctor}

CLINICAL HISTORY: Routine screening mammography

TECHNIQUE: {scan_type.title()} with standard CC and MLO views of the {laterality} breast(s).

FINDINGS: {findings.capitalize()}. Breast composition shows scattered fibroglandular densities.

IMPRESSION: {birads}
RECOMMENDATION: {follow_up}

Patient contact information: {phone}, {email}
Residence: {address}
Originally from: {patient_location}
Account #: {account_number}
        """,
        
        # Template 2: Narrative format
        f"""
On {scan_date}, patient {name} (born {dob}) presented to {hospital_name} located at {hospital_location} for {scan_type}. 
The examination was performed and interpreted by Dr. {doctor}. Imaging of the {laterality} breast revealed {findings}. 
The breast tissue demonstrated normal density patterns. Assessment: {birads}. 
Clinical recommendation: {follow_up}
Patient identifiers: MRN {mrn}, SSN {ssn}, Study ID {study_id}.
Contact: {phone} / {email}. Current address: {address}, hometown: {patient_location}.
        """,
        
        # Template 3: Structured clinical note
        f"""
BREAST IMAGING REPORT
Date of Service: {scan_date}
Patient Name: {name}
Date of Birth: {dob}
Medical Record Number: {mrn}
Patient ID: {patient_id}
Facility: {hospital_name}
Address: {hospital_location}
Radiologist: Dr. {doctor}
Procedure: {scan_type.title()}
Laterality: {laterality.title()}
Results: {findings.capitalize()}
BI-RADS Category: {birads}
Follow-up: {follow_up}
Patient Demographics: Resides at {address}, originally from {patient_location}
Emergency Contact: {phone}, {email}
Case Number: {case_number}
        """,
        
        # Template 4: Brief clinical summary
        f"""
{name} (DOB: {dob}, MRN: {mrn}) underwent {scan_type} on {scan_date} at {hospital_name}, {hospital_location}. 
Dr. {doctor} reviewed the {laterality} breast images showing {findings}. 
Classification: {birads}. Plan: {follow_up}
Demographics: {address}, from {patient_location}
Contact: {phone}, {email}
Reference: Account #{account_number}, Certificate #{certificate_number}
        """,
        
        # Template 5: Detailed assessment format
        f"""
MAMMOGRAPHIC EXAMINATION REPORT
Examination performed on {scan_date} for {name} (date of birth {dob})
Location: {hospital_name} - {hospital_location}
Interpreting radiologist: Dr. {doctor}
Medical identifiers: MRN {mrn}, Health Plan ID {health_plan_id}, License #{license_number}
Procedure: {scan_type.title()} examination of {laterality} breast
Clinical findings: {findings.capitalize()}
Breast density: Within normal limits
BI-RADS assessment: {birads}
Management recommendation: {follow_up}
Patient residence: {address}
Originally from: {patient_location}
Preferred contact: {phone} or {email}
        """,
        
        # Template 6: Abbreviated report style
        f"""
{hospital_name} ({hospital_location}) - MAMMOGRAPHY RESULTS
Patient: {name}, DOB {dob}
Date: {scan_date}
Physician: Dr. {doctor}
Type: {scan_type.title()} - {laterality} breast
Findings: {findings.capitalize()}
Rating: {birads}
Next steps: {follow_up}
Patient info: {address}, originally {patient_location}
Contact: {phone}, {email}
IDs: MRN {mrn}, Account {account_number}, Device {device_serial}
        """,
        
        # Template 7: Comprehensive clinical documentation
        f"""
RADIOLOGY REPORT - MAMMOGRAPHY
Patient Information:
Name: {name}
Birth Date: {dob}
Medical Record: {mrn}
Social Security: {ssn}
Study Date: {scan_date}
Institution: {hospital_name}
Location: {hospital_location}
Attending Physician: Dr. {doctor}
Examination Type: {scan_type.title()}
Breast(s) Examined: {laterality.title()}
Interpretation: {findings.capitalize()}
Breast Composition: Normal
BI-RADS Category: {birads}
Recommended Follow-up: {follow_up}
Patient Address: {address}
Hometown: {patient_location}
Phone: {phone}
Email: {email}
Additional Identifiers: Vehicle ID {vehicle_id}, License Plate {license_plate}
        """,
        
        # Template 8: Hospital system format
        f"""
{hospital_name} - BREAST IMAGING SERVICES
{hospital_location}
PATIENT: {name} (DOB: {dob})
EXAM DATE: {scan_date}
RADIOLOGIST: Dr. {doctor}
STUDY TYPE: {scan_type.title()}
LATERALITY: {laterality.title()}
MEDICAL RECORD: {mrn}
PATIENT ACCOUNT: {account_number}
FINDINGS: {findings.capitalize()}
ASSESSMENT: {birads}
RECOMMENDATIONS: {follow_up}
PATIENT DEMOGRAPHICS:
Current Address: {address}
Place of Origin: {patient_location}
Telephone: {phone}
Email Address: {email}
Additional IDs: Certificate #{certificate_number}, Study #{study_id}
        """,
        
        # Template 9: Quick reference format
        f"""
MAMMOGRAM RESULTS - {scan_date}
Pt: {name} (born {dob}) | MRN: {mrn} | ID: {patient_id}
Facility: {hospital_name}, {hospital_location}
Radiologist: Dr. {doctor}
Exam: {scan_type.title()} ({laterality})
Results: {findings.capitalize()}
Score: {birads}
Action: {follow_up}
Address: {address} (from {patient_location})
Contact: {phone} / {email}
Ref: Health Plan {health_plan_id}, License {license_number}
        """,
        
        # Template 10: Formal medical report
        f"""
DEPARTMENT OF RADIOLOGY
{hospital_name}
{hospital_location}

MAMMOGRAPHY REPORT

PATIENT DEMOGRAPHICS:
Full Name: {name}
Date of Birth: {dob}
Medical Record Number: {mrn}
Social Security Number: {ssn}
Current Address: {address}
City of Origin: {patient_location}
Contact Phone: {phone}
Email: {email}

EXAMINATION DETAILS:
Date of Examination: {scan_date}
Procedure: {scan_type.title()}
Breast(s) Examined: {laterality.title()}
Interpreting Physician: Dr. {doctor}

FINDINGS AND IMPRESSION:
Imaging demonstrates {findings}. Breast parenchyma shows normal density.
BI-RADS Classification: {birads}
Clinical Recommendation: {follow_up}

PATIENT IDENTIFIERS:
Account Number: {account_number}
Case Number: {case_number}
Device Serial: {device_serial}
        """
    ]

    template = random.choice(templates).strip()

    # Annotate PHI spans with the 6 entity categories
    entities = []
    for ent_text, label in [
        (name, "NAME"),
        (doctor, "NAME"),
        (dob, "DATE"),
        (scan_date, "DATE"),
        (address, "LOCATION"),
        (hospital_location, "LOCATION"),
        (patient_location, "LOCATION"),
        (hospital_name, "HOSPITAL"),
        (mrn, "ID"),
        (ssn, "ID"),
        (health_plan_id, "ID"),
        (account_number, "ID"),
        (license_number, "ID"),
        (certificate_number, "ID"),
        (license_plate, "ID"),
        (vehicle_id, "ID"),
        (device_serial, "ID"),
        (patient_id, "ID"),
        (case_number, "ID"),
        (study_id, "ID"),
        (phone, "CONTACT"),
        (email, "CONTACT"),
    ]:  
        start = template.find(ent_text)
        if start != -1:
            end = start + len(ent_text)
            entities.append({"start": start, "end": end, "label": label, "text": ent_text})

    return {"text": template, "entities": entities}



"""
PROMPT 2:
Here is a CT Scan report example:

Example 1:
Patient: DOE, JOHN Referring Physician: DR. DAVID LIVESEY MRN : JD4USARAD DOB: 01/01/1961 Exam Date: 06/05/2010 FAX: (305) 418-8166 CLINICAL HISTORY: Loss of balance. TECHNIQUE: Multiple axial CT images were obtained through the brain without IV contrast material. COMMENTS: The study shows normal configuration of sella turcica. There are no intra or extra-axial collections. There is no mass effect or midline shift. There is no evidence of hematoma formation. No hydrocephalus is present. The ventricles are symmetrical. No abnormal calcifications are present. Changes of diffuse age-appropriate cerebellar and cerebral atrophy are noted with proportionally dilated ventricles and cortical sulci. There are bilateral periventricular white matter hypolucencies compatible with chronic microvascular disease. Otherwise, no significant focal abnormalities are seen either in the posterior fossa or supratentorial compartment. There has been no significant interval change since a prior examination dated 3/29/10. IMPRESSION: 1. Age-appropriate cerebellar and cerebral atrophy. 2. Chronic periventricular white matter microvascular disease 3. No CT evidence of acute intra-axial pathology.

Example 2:
2601 E. Oakland Park Blvd, Suite 102 Ft. Lauderdale, FL 33306 Phone: 888.886.5238 Fax: 888.886.5221 Patient: DOE, JOHN Referring Physician: DR. DAVID LIVESEY MRN : JD4USARAD DOB: 01/01/1961 Exam Date: 06/05/2010 FAX: (305) 418-8166 IMPRESSION: ABDOMEN: 1. Several fluid-filled loops of small bowel are present compatible with mild enteritis. 2. Fatty liver. 3. Status post cholecystectomy. 4. Fat containing umbilical hernia. PELVIS: 1. No evidence of diverticulitis or acute inflammatory process in the pelvis. Discussed with Dr. DAVID LIVESEY -Electronically Signed by: RADIOLOGIST, ADMIN on 06/05/2010 2:31:33 PM

Here is what I am trying to do:
1. Create a list of templates for different paraphrased ct scan reports
2. Each of these reports must focus on including the following 6 entities: 1. Name 2. Date 3. Location 4. Hospital 5. Identifier (can be: Medical record number, patient number, etc) 6. Contact
3. Objective: Create synthetic medical reports in a way that I am able to extract synthetic entities (6 entities) from it to train a language model to be a named entity recognition model
4. Your goal: synthesize the data like an expert data synthesizer

Example of previous templates (these are just examples, you dont have to duplicate):
"""


def generate_ct_scan_report():
    imaging_centers = [
        ("City Medical Imaging Center", "123 Radiology St, Metropolis, State"),
        ("Westside Diagnostic Imaging", "456 Scan Blvd, Townsville, State"),
        ("Eastside Imaging Facility", "789 CT Rd, Villagetown, State"),
        ("Central Imaging Clinic", "101 Xray Ave, Metrocity, State"),
        ("Regional Medical Imaging", "555 Health Plaza, Riverside, State"),
        ("Advanced Diagnostic Center", "777 Medical Dr, Northpoint, State"),
        ("Community Radiology Services", "999 Wellness Blvd, Southside, State"),
        ("Metropolitan CT Imaging", "321 University Ave, Centraltown, State"),
        ("Precision Imaging Institute", "654 Technology Way, Westbrook, State"),
        ("Harbor Medical Imaging", "987 Coastal Rd, Eastwood, State")
    ]

    patient_locations = [
        "Metropolis", "Townsville", "Villagetown", "Metrocity",
        "Lakeside", "Hilltown", "Riverside", "Northpoint",
        "Southside", "Centraltown", "Westbrook", "Eastwood"
    ]

    imaging_center, center_location = random.choice(imaging_centers)
    patient_location = random.choice(patient_locations)
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=20, maximum_age=90).strftime("%B %d, %Y")
    address = fake.address().replace("\n", ", ")
    doctor = fake.name()
    referring_doctor = fake.name()
    phone = fake.phone_number()
    email = fake.email()
    fax = fake.phone_number()
    ssn = fake.ssn()
    health_plan_id = fake.bothify(text="HP-########")
    account_number = fake.bothify(text="ACCT-#######")
    license_number = fake.bothify(text="LIC-#######")
    certificate_number = fake.bothify(text="CERT-#####")
    patient_id = fake.bothify(text="PT-####")
    case_number = fake.bothify(text="CS-######")
    study_id = fake.bothify(text="STU-#####")
    accession_number = fake.bothify(text="ACC-######")
    mrn = fake.bothify(text="#####")
    scan_date = fake.date()
    
    scan_type = random.choice([
        "non-contrast CT scan", "contrast-enhanced CT scan", "CT with IV contrast",
        "CT without contrast", "multiplanar CT imaging", "helical CT scan"
    ])
    
    body_part = random.choice([
        "head", "chest", "abdomen", "pelvis", "cervical spine", 
        "lumbar spine", "thoracic spine", "abdomen and pelvis"
    ])
    
    indication = random.choice([
        "trauma evaluation", "suspected infection", "cancer staging",
        "acute abdominal pain", "follow-up of known pathology",
        "headache evaluation", "chest pain workup", "back pain assessment",
        "screening for malignancy", "post-operative evaluation"
    ])
    
    findings = random.choice([
        "no acute intracranial hemorrhage or mass effect",
        "small right lower lobe pneumonia",
        "no evidence of bowel obstruction or free air",
        "mild hepatomegaly with no focal lesions",
        "enlarged lymph nodes in the mediastinum",
        "degenerative changes in the lumbar spine",
        "no acute osseous abnormalities",
        "normal brain parenchyma without mass lesion",
        "unremarkable abdominal viscera",
        "bilateral pleural effusions, small"
    ])
    
    impression = random.choice([
        "normal study", "findings consistent with pneumonia",
        "no acute abdominal pathology detected",
        "hepatomegaly likely secondary to fatty infiltration",
        "lymphadenopathy requires further evaluation",
        "degenerative spine disease without acute findings",
        "no evidence of acute intracranial pathology",
        "bilateral pleural effusions, likely inflammatory",
        "unremarkable CT examination",
        "mild age-related changes without acute abnormality"
    ])

    templates = [
        # Template 1: Traditional typed report with header letterhead
        f"""
                    {imaging_center}
                    {center_location}
              Phone: {phone} | Fax: {fax}

COMPUTED TOMOGRAPHY EXAMINATION REPORT

Patient Name: {name}                    Medical Record #: {mrn}
Date of Birth: {dob}                   Examination Date: {scan_date}
Referring Physician: Dr. {referring_doctor}
Interpreting Radiologist: Dr. {doctor}

CLINICAL INDICATION: {indication.capitalize()}

TECHNIQUE AND FINDINGS:
{scan_type.capitalize()} imaging of the {body_part} demonstrates {findings}.

RADIOLOGIC IMPRESSION:
{impression.capitalize()}

PATIENT CONTACT INFORMATION:
Residence: {address}
Originally from: {patient_location}
Telephone: {phone}
Electronic mail: {email}
Account Reference: {account_number}
        """,
        
        # Template 2: Modern electronic health record format
        f"""

                    RADIOLOGY REPORT                         
                 {imaging_center}                
                 {center_location}               

>>> PATIENT DEMOGRAPHICS <<<
NAME: {name}
DOB: {dob}
MRN: {mrn}
SSN: {ssn}
PATIENT ID: {patient_id}

>>> STUDY DETAILS <<<
EXAM DATE: {scan_date}
STUDY TYPE: {scan_type.upper()}
BODY REGION: {body_part.upper()}
ACCESSION #: {accession_number}

>>> CLINICAL TEAM <<<
ORDERING PROVIDER: Dr. {referring_doctor}
INTERPRETING RADIOLOGIST: Dr. {doctor}
INDICATION: {indication.capitalize()}

>>> RESULTS <<<
FINDINGS: {findings.capitalize()}
IMPRESSION: {impression.capitalize()}

>>> CONTACT INFO <<<
HOME ADDRESS: {address}
HOMETOWN: {patient_location}
PHONE: {phone} | EMAIL: {email}
INSURANCE: {health_plan_id}
        """,
        
        # Template 3: Abbreviated shorthand clinical note
        f"""
CT-{body_part.upper()[:3]} / {scan_date} / {imaging_center}

Pt: {name} ({dob}) MRN#{mrn}
Loc: {center_location}
Ref: Dr.{referring_doctor} → Rad: Dr.{doctor}
Hx: {indication}

Study: {scan_type}
Findings: {findings}
Dx: {impression}

Contact: {phone}/{email}
Address: {address} (orig: {patient_location})
Case#: {case_number} | Study#: {study_id}
        """,
        
        # Template 4: Detailed academic hospital narrative
        f"""
DEPARTMENT OF RADIOLOGY AND IMAGING SCIENCES
{imaging_center}
{center_location}
Telephone: {phone} | Electronic correspondence: {email}

═══════════════════════════════════════════════════════════════════

COMPUTED TOMOGRAPHY CONSULTATION REPORT

This is to certify that on {scan_date}, {name} (date of birth: {dob}) presented for computed tomographic evaluation of the {body_part} region. The patient, who maintains residence at {address} and originates from {patient_location}, was referred by Dr. {referring_doctor} with the clinical indication of {indication}.

Following careful review of the {scan_type} images by the undersigned radiologist, Dr. {doctor}, the following observations were documented: {findings}. 

Based upon these radiographic findings, the clinical impression rendered is as follows: {impression}.

The patient may be contacted via telephone at {phone} or electronic mail at {email}. Medical record identifiers include MRN {mrn}, Social Security Number {ssn}, and Health Plan Beneficiary ID {health_plan_id}.

Respectfully submitted,
Dr. {doctor}, M.D.
Board-Certified Radiologist
        """,
        
        # Template 5: International medical center format with codes
        f"""
◊◊◊ INTERNATIONAL IMAGING CONSORTIUM ◊◊◊
{imaging_center}
{center_location}

---RADIOLOGY TRANSMISSION---
DATE: {scan_date}
PROTOCOL: CT-{body_part.upper()[:4]}
PATIENT: {name}
D.O.B: {dob}
REF.MD: {referring_doctor}
RADIOLOGIST: {doctor}

MEDICAL RECORD NO: {mrn}
CERTIFICATE NO: {certificate_number}
LICENSE ID: {license_number}

EXAMINATION PARAMETERS:
→ Modality: {scan_type.capitalize()}
→ Region: {body_part.capitalize()}  
→ Clinical Context: {indication.capitalize()}

RADIOLOGICAL ASSESSMENT:
→ Findings: {findings.capitalize()}
→ Interpretation: {impression.capitalize()}

PATIENT REGISTRY:
→ Current Address: {address}
→ Place of Birth: {patient_location}
→ Primary Contact: {phone}
→ Email Registry: {email}
→ Account Designation: {account_number}

---END TRANSMISSION---
        """,
        
        # Template 6: Emergency department stat report
        f"""
STAT RADIOLOGY REPORT 
{imaging_center} - EMERGENCY SERVICES
{center_location}

⚡ URGENT STUDY COMPLETED: {scan_date} ⚡

PATIENT ALERT: {name}
BORN: {dob} | RECORD: {mrn} | SSN: {ssn}
EMERGENCY CONTACT: {phone}
CURRENT ADDRESS: {address}
ORIGIN: {patient_location}

EMERGENCY PHYSICIAN: Dr. {referring_doctor}  
RADIOLOGIST ON CALL: Dr. {doctor}

SCAN TYPE: {scan_type.upper()} - {body_part.upper()}
CLINICAL REASON: {indication.upper()}

⚡ CRITICAL FINDINGS ⚡
{findings.upper()}

⚡ EMERGENCY IMPRESSION ⚡  
{impression.upper()}

NOTIFY: {email}
ACCOUNT: {account_number} | STUDY: {study_id}
        """,
        
        # Template 7: Minimalist bullet-point format
        f"""
{imaging_center} | {center_location}

• PATIENT: {name}
• DOB: {dob} 
• MRN: {mrn}
• SCAN DATE: {scan_date}
• STUDY: {scan_type} → {body_part}
• ORDERED BY: Dr. {referring_doctor}
• READ BY: Dr. {doctor}
• REASON: {indication}

• FINDINGS: {findings}
• IMPRESSION: {impression}

• ADDRESS: {address}
• FROM: {patient_location}  
• PHONE: {phone}
• EMAIL: {email}
• ID: {patient_id} | ACC: {accession_number}
        """,
        
        # Template 8: Formal hospital letterhead documentation
        f"""
                        OFFICE OF DIAGNOSTIC RADIOLOGY
                            {imaging_center}
                            {center_location}
                        
                    Official Medical Correspondence
                    
    ═══════════════════════════════════════════════════════════════

Dear Dr. {referring_doctor},

Re: Computed Tomography Examination - {name}

I am writing to provide you with the results of the {scan_type} examination performed on {scan_date} for your patient, {name}, born {dob}.

The patient, who resides at {address} and is originally from {patient_location}, underwent imaging of the {body_part} for the indication of {indication}.

Upon careful review of the study, I observed the following: {findings}.

My radiological interpretation is: {impression}.

Should you require any clarification regarding these findings, please do not hesitate to contact me directly. The patient may be reached at {phone} or {email}.

Medical Record Number: {mrn}
Social Security Number: {ssn}  
Health Plan Identifier: {health_plan_id}
License Number: {license_number}

Sincerely yours,

Dr. {doctor}, M.D.
Diagnostic Radiologist
        """]
    template = random.choice(templates).strip()

    # Annotate PHI spans with the 6 entity categories
    entities = []
    for ent_text, label in [
        (name, "NAME"),
        (doctor, "NAME"),
        (referring_doctor, "NAME"),
        (dob, "DATE"),
        (scan_date, "DATE"),
        (address, "LOCATION"),
        (center_location, "LOCATION"),
        (patient_location, "LOCATION"),
        (imaging_center, "HOSPITAL"),
        (mrn, "ID"),
        (ssn, "ID"),
        (health_plan_id, "ID"),
        (account_number, "ID"),
        (license_number, "ID"),
        (certificate_number, "ID"),
        (patient_id, "ID"),
        (case_number, "ID"),
        (study_id, "ID"),
        (accession_number, "ID"),
        (phone, "CONTACT"),
        (email, "CONTACT"),
        (fax, "CONTACT"),
    ]:  
        start = template.find(ent_text)
        if start != -1:
            end = start + len(ent_text)
            entities.append({"start": start, "end": end, "label": label, "text": ent_text})

    return {"text": template, "entities": entities}


def generate_admission_discharge_dates():
    admission = fake.date_between(start_date='-30d', end_date='-1d')
    discharge = fake.date_between(start_date=admission, end_date='today')
    return admission.strftime("%B %d, %Y"), discharge.strftime("%B %d, %Y")

def generate_discharge_summary():
    # Generate synthetic data
    name = fake.name()
    doctor = f"Dr. {fake.last_name()}"
    referring_doctor = f"Dr. {fake.last_name()}"
    dob = fake.date_of_birth(minimum_age=18, maximum_age=95).strftime("%m/%d/%Y")
    scan_date = fake.date_between(start_date='-7d', end_date='today').strftime("%m/%d/%Y")
    address = fake.address().replace("\n", ", ")
    phone = fake.phone_number()
    email = fake.email()
    fax = fake.bothify(text='(###) ###-####')
    
    # Generate various identifier formats
    mrn = f"MRN: {fake.bothify(text='######')}"
    patient_id = f"Patient ID: {fake.bothify(text='??####')}"
    medical_record = f"Medical Record #: {fake.bothify(text='#######')}"
    hospital_id = f"Hospital ID: {fake.bothify(text='H###-####')}"
    account_number = f"Account Number: {fake.bothify(text='ACC-#######')}"
    ssn = fake.ssn()
    health_plan_id = fake.bothify(text='HP#######')
    license_number = fake.bothify(text='LIC######')
    certificate_number = fake.bothify(text='CERT#####')
    case_number = fake.bothify(text='CASE#####')
    study_id = fake.bothify(text='STU######')
    accession_number = fake.bothify(text='ACC######')
    
    identifiers = [mrn, patient_id, medical_record, hospital_id, account_number]
    identifier = random.choice(identifiers)
    
    hospitals = [
        "St. Mary's Medical Center",
        "General Hospital",
        "University Medical Center", 
        "Regional Medical Center",
        "Community Hospital",
        "Memorial Healthcare System",
        "Sacred Heart Hospital",
        "Presbyterian Medical Center"
    ]
    
    locations = [
        "Internal Medicine Unit, 3rd Floor",
        "Cardiac Care Unit",
        "Medical/Surgical Floor 4B",
        "Progressive Care Unit",
        "Orthopedic Ward",
        "Pulmonary Unit, Building A",
        "Emergency Department Observation"
    ]
    
    # Additional location entities
    center_location = fake.city() + " Medical District"
    patient_location = fake.address().replace("\n", ", ")
    
    diagnoses = [
        "acute exacerbation of chronic heart failure",
        "community-acquired pneumonia with respiratory distress", 
        "acute coronary syndrome with ST elevation",
        "diabetic ketoacidosis with dehydration",
        "post-operative complications following hip replacement",
        "acute kidney injury secondary to dehydration",
        "cerebrovascular accident with left-sided weakness"
    ]
    
    treatments = [
        "diuretic therapy and cardiac monitoring",
        "broad-spectrum antibiotics and respiratory support",
        "emergency cardiac catheterization with stent placement",
        "insulin therapy and electrolyte correction",
        "surgical revision and antibiotic prophylaxis",
        "IV fluid resuscitation and nephrology consultation",
        "thrombolytic therapy and neurological rehabilitation"
    ]
    
    # Select random values
    admission_date, discharge_date = generate_admission_discharge_dates()
    hospital = random.choice(hospitals)
    location = random.choice(locations)
    diagnosis = random.choice(diagnoses)
    treatment = random.choice(treatments)
    
    # Template variations with diverse entity inclusion
    templates = [
        # Template 1 - Formal structured format
        f"""DISCHARGE SUMMARY
        
Patient: {name}
{identifier}
Date of Birth: {dob}
Hospital: {hospital}
Location: {location}
Contact: {phone}

ADMISSION DATE: {admission_date}
DISCHARGE DATE: {discharge_date}

PRINCIPAL DIAGNOSIS: {diagnosis.title()}

HOSPITAL COURSE:
The patient was admitted to {hospital} on {admission_date} presenting with {diagnosis}. During the hospitalization at {location}, the patient received {treatment}. Clinical improvement was noted throughout the stay. The patient was deemed stable for discharge on {discharge_date}.

DISCHARGE INSTRUCTIONS:
Follow-up appointment scheduled. Contact number on file: {phone}. Return if symptoms worsen.""",

        # Template 2 - Narrative style
        f"""Mr./Ms. {name} ({identifier}) was admitted to {hospital}, {location} on {admission_date} for management of {diagnosis}. The patient's emergency contact ({phone}) was notified upon admission. 

Throughout the hospital stay, {treatment} was provided with excellent clinical response. The patient remained stable and was successfully discharged on {discharge_date}. 

Discharge planning included coordination with outpatient services. The patient was advised to maintain contact via {phone} for any concerns.""",

        # Template 3 - Bullet point format
        f"""PATIENT INFORMATION:
• Name: {name}
• Hospital: {hospital}
• Unit: {location}  
• {identifier}
• Contact: {phone}
• DOB: {dob}

ADMISSION/DISCHARGE DATES:
• Admitted: {admission_date}
• Discharged: {discharge_date}

CLINICAL SUMMARY:
Patient presented with {diagnosis} and received {treatment}. Recovery was uncomplicated. Discharged home in stable condition with follow-up arrangements.""",

        # Template 4 - Brief clinical note style
        f"""{name} | {identifier} | {hospital}
Location: {location} | Contact: {phone}

Admission: {admission_date} → Discharge: {discharge_date}

DIAGNOSIS: {diagnosis.upper()}
TREATMENT: {treatment.title()}

Patient tolerated treatment well. Discharged in improved condition. Follow-up as arranged.""",

        # Template 5 - Detailed narrative with additional entities
        f"""DISCHARGE SUMMARY FOR {name.upper()}

This {fake.random_int(min=25, max=85)}-year-old patient ({identifier}) was admitted to {hospital}, specifically to {location}, on {admission_date}. Patient address: {address}. 

ATTENDING PHYSICIAN: {doctor}
REFERRING PHYSICIAN: {referring_doctor}

PRESENTING CONDITION: {diagnosis.title()}

CLINICAL COURSE: The patient underwent {treatment} with monitoring by our clinical team. Progressive improvement was documented throughout the admission period.

DISCHARGE: Patient was discharged on {discharge_date} in stable condition. Emergency contact information ({phone}) has been updated in the medical record.

SSN: {ssn} | Case Number: {case_number}
All discharge medications and follow-up instructions were reviewed with the patient.""",

        # Template 6 - Administrative focus
        f"""FACILITY: {hospital}
PATIENT: {name} ({identifier})
WARD: {location}
EMERGENCY CONTACT: {phone}
EMAIL: {email}

DATE RANGE: {admission_date} through {discharge_date}

PRIMARY DIAGNOSIS: {diagnosis}
INTERVENTION: {treatment}

Patient demonstrated appropriate response to therapy. Discharge criteria met on {discharge_date}. Post-discharge care coordinated.
Health Plan ID: {health_plan_id} | Study ID: {study_id}""",

        # Template 7 - Chronological format
        f"""PATIENT CARE SUMMARY

{name} (DOB: {dob}, {identifier})
Treated at: {hospital} - {location}
Contact Information: {phone}
Patient Address: {patient_location}

TIMELINE:
{admission_date}: Patient admitted for {diagnosis}
{scan_date}: Diagnostic imaging completed
Hospital Course: {treatment} initiated with good tolerance
{discharge_date}: Discharged home in stable condition

Attending: {doctor} | License: {license_number}
Follow-up care arrangements confirmed prior to discharge.""",

        # Template 8 - Problem-oriented format
        f"""PROBLEM-ORIENTED DISCHARGE SUMMARY

Patient Details: {name}, {identifier}
Facility: {hospital}, {location}
Phone: {phone} | Fax: {fax}
Service Dates: {admission_date} to {discharge_date}

PROBLEM: {diagnosis.title()}
INTERVENTION: {treatment.title()}
OUTCOME: Successful treatment with clinical improvement

Patient education completed. Discharge planning coordinated with family via {phone}.
Accession Number: {accession_number}""",

        # Template 9 - Concise format
        f"""{hospital} - {location}
Patient: {name} | {identifier} | {phone}

{admission_date} - {discharge_date}

Admitted with: {diagnosis}
Treated with: {treatment}
Outcome: Stable discharge

Hospital Location: {center_location}
Post-acute care arrangements finalized.""",

        # Template 10 - Comprehensive format
        f"""INPATIENT DISCHARGE DOCUMENTATION

PATIENT IDENTIFICATION:
Name: {name}
Medical Record: {identifier}
Date of Birth: {dob}
Primary Contact: {phone}
Email: {email}
Address: {address}

FACILITY INFORMATION:
Hospital: {hospital}
Care Unit: {location}

MEDICAL TEAM:
Attending: {doctor}
Referring: {referring_doctor}

ADMISSION/DISCHARGE:
Admitted: {admission_date}
Discharged: {discharge_date}
Length of Stay: {fake.random_int(min=1, max=14)} days

CLINICAL SUMMARY:
Primary Diagnosis: {diagnosis}
Treatment Protocol: {treatment}
Clinical Response: Favorable
Discharge Status: Stable

ADMINISTRATIVE:
SSN: {ssn}
Certificate Number: {certificate_number}
The patient received comprehensive care during this admission. All discharge requirements have been satisfied."""
    ]
    
    template = random.choice(templates)
    
    # Annotate PHI spans with the 6 entity categories
    entities = []
    for ent_text, label in [
        (name, "NAME"),
        (doctor, "NAME"),
        (referring_doctor, "NAME"),
        (dob, "DATE"),
        (scan_date, "DATE"),
        (admission_date, "DATE"),
        (discharge_date, "DATE"),
        (address, "LOCATION"),
        (center_location, "LOCATION"),
        (patient_location, "LOCATION"),
        (location, "LOCATION"),
        (hospital, "HOSPITAL"),
        (identifier, "ID"),
        (ssn, "ID"),
        (health_plan_id, "ID"),
        (license_number, "ID"),
        (certificate_number, "ID"),
        (case_number, "ID"),
        (study_id, "ID"),
        (accession_number, "ID"),
        (phone, "CONTACT"),
        (email, "CONTACT"),
        (fax, "CONTACT"),
    ]:  
        start = template.find(ent_text)
        if start != -1:
            end = start + len(ent_text)
            entities.append({"start": start, "end": end, "label": label, "text": ent_text})

    return {"text": template, "entities": entities}
    

"""
PROMPT 4:
Here is an example of an x ray report, 

I am trying to do the following:

Create a list of templates for different paraphrased mammography reports Each of these reports must focus on including the following 6 entities: 1. Name 2. Date 3. Location 4. Hospital 5. Identifier (can be: Medical record number, patient number, etc) 6. Contact Objective: Create synthetic medical reports in a way that I am able to extract synthetic entities (6 entities) from it to train a language model to be a named entity recognition model Your goal: synthesize the data like an expert data synthesizer like an actual clinican from the hospital:

here is an example of what i did for generating mammography repirts:

make sure each report is distinctive enough but good enough to be accurate

"""
def generate_xray_report():
    hospitals = [
        ("City General Hospital", "123 Health St, Cityville, State"),
        ("Westside Medical Center", "456 Care Ave, Townsville, State"),
        ("Eastside Radiology Clinic", "789 Wellness Rd, Villagetown, State"),
        ("Central Imaging Institute", "101 Recovery Blvd, Metrocity, State"),
        ("Regional Diagnostic Center", "555 Healing Ln, Riverside, State"),
        ("Metropolitan Radiology Unit", "777 Therapy St, Centraltown, State"),
        ("Community Imaging Services", "999 Clinic Dr, Southside, State"),
        ("Advanced Radiology Institute", "321 Treatment Ave, Northville, State"),
    ]

    patient_locations = [
        "Cityville", "Townsville", "Villagetown", "Metrocity",
        "Lakeside", "Hilltown", "Riverside", "Centraltown",
        "Southside", "Northville", "Westbrook", "Eastwood"
    ]

    hospital_name, hospital_location = random.choice(hospitals)
    patient_location = random.choice(patient_locations)
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%B %d, %Y")
    address = fake.address().replace("\n", ", ")
    doctor = fake.name()
    phone = fake.phone_number()
    email = fake.email()
    mrn = fake.bothify(text="#####")
    ssn = fake.ssn()
    health_plan_id = fake.bothify(text="########")
    account_number = fake.bothify(text="#######")
    license_number = fake.bothify(text="#######")
    certificate_number = fake.bothify(text="#####")
    license_plate = fake.bothify(text="???###")
    vehicle_id = fake.bothify(text="########")
    device_serial = fake.bothify(text="DEV-SN-######")
    patient_id = fake.bothify(text="PT-####")
    case_number = fake.bothify(text="CN-######")
    study_id = fake.bothify(text="STU-#####")
    scan_date = fake.date()
    scan_type = random.choice(["chest X-ray", "abdominal X-ray", "thoracic spine X-ray", "lumbar spine X-ray", "rib X-ray"])
    laterality = random.choice(["left", "right", "bilateral"])

    findings = random.choice([
        "no radiographic evidence for acute fracture, dislocation or focal soft tissue abnormality",
        "diffuse thoracic disc degeneration and spondylosis",
        "no evidence for a displaced fracture",
        "diffuse degenerative disc disease without evidence for fracture or dislocation",
        "levoscoliosis present",
        "multiple radiographic views demonstrate no acute fracture",
        "focal soft tissue pathology identified",
        "evidence of diffuse degenerative changes",
        "no substantive interval change from previous study",
        "well-circumscribed mass likely benign"
    ])

    impression = random.choice([
        "No acute fracture. Diffuse degenerative changes noted.",
        "No evidence for a displaced fracture. Consider bone scan if further concern.",
        "Diffuse degenerative disc disease with levoscoliosis.",
        "No acute fracture. Thoracic disc degeneration present.",
        "No radiographic evidence of acute pathology.",
        "Diffuse spondylosis noted without acute fracture.",
        "Degenerative changes present without dislocation.",
        "Stable findings compared to previous study.",
        "Benign appearance of identified mass.",
        "Further evaluation recommended for identified pathology."
    ])

    follow_up = random.choice([
        "Annual follow-up recommended.",
        "Ultrasound follow-up in 6 months.",
        "Biopsy recommended for further evaluation.",
        "Short interval follow-up advised.",
        "Continue routine monitoring.",
        "Diagnostic imaging in 12 months.",
        "Immediate surgical consultation required.",
        "MRI correlation suggested."
    ])

    templates = [
        # Template 1: Standard clinical format
        f"""
X-RAY REPORT
Patient: {name}
DOB: {dob}
MRN: {mrn}
Study Date: {scan_date}
Facility: {hospital_name}, {hospital_location}
Interpreting Physician: Dr. {doctor}

CLINICAL HISTORY: Routine {scan_type}

TECHNIQUE: {scan_type.title()} with standard views of the {laterality} region.

FINDINGS: {findings.capitalize()}.

IMPRESSION: {impression}
RECOMMENDATION: {follow_up}

Patient contact information: {phone}, {email}
Residence: {address}
Originally from: {patient_location}
Account #: {account_number}
        """,

        # Template 2: Narrative format
        f"""
On {scan_date}, patient {name} (born {dob}) presented to {hospital_name} located at {hospital_location} for {scan_type}.
The examination was performed and interpreted by Dr. {doctor}. Imaging of the {laterality} region revealed {findings}.
Assessment: {impression}.
Clinical recommendation: {follow_up}
Patient identifiers: MRN {mrn}, SSN {ssn}, Study ID {study_id}.
Contact: {phone} / {email}. Current address: {address}, hometown: {patient_location}.
        """,

        # Template 3: Structured clinical note
        f"""
RADIOLOGY REPORT - X-RAY
Date of Service: {scan_date}
Patient Name: {name}
Date of Birth: {dob}
Medical Record Number: {mrn}
Patient ID: {patient_id}
Facility: {hospital_name}
Address: {hospital_location}
Radiologist: Dr. {doctor}
Procedure: {scan_type.title()}
Laterality: {laterality.title()}
Results: {findings.capitalize()}
Impression: {impression}
Follow-up: {follow_up}
Patient Demographics: Resides at {address}, originally from {patient_location}
Emergency Contact: {phone}, {email}
Case Number: {case_number}
        """,

        # Template 4: Brief clinical summary
        f"""
{name} (DOB: {dob}, MRN: {mrn}) underwent {scan_type} on {scan_date} at {hospital_name}, {hospital_location}.
Dr. {doctor} reviewed the {laterality} region images showing {findings}.
Classification: {impression}. Plan: {follow_up}
Demographics: {address}, from {patient_location}
Contact: {phone}, {email}
Reference: Account #{account_number}, Certificate #{certificate_number}
        """,

        # Template 5: Detailed assessment format
        f"""
X-RAY EXAMINATION REPORT
Examination performed on {scan_date} for {name} (date of birth {dob})
Location: {hospital_name} - {hospital_location}
Interpreting radiologist: Dr. {doctor}
Medical identifiers: MRN {mrn}, Health Plan ID {health_plan_id}, License #{license_number}
Procedure: {scan_type.title()} examination of {laterality} region
Clinical findings: {findings.capitalize()}
Impression: {impression}
Management recommendation: {follow_up}
Patient residence: {address}
Originally from: {patient_location}
Preferred contact: {phone} or {email}
        """,

        # Template 6: Abbreviated report style
        f"""
{hospital_name} ({hospital_location}) - X-RAY RESULTS
Patient: {name}, DOB {dob}
Date: {scan_date}
Physician: Dr. {doctor}
Type: {scan_type.title()} - {laterality} region
Findings: {findings.capitalize()}
Impression: {impression}
Next steps: {follow_up}
Patient info: {address}, originally {patient_location}
Contact: {phone}, {email}
IDs: MRN {mrn}, Account {account_number}, Device {device_serial}
        """,

        # Template 7: Comprehensive clinical documentation
        f"""
RADIOLOGY REPORT - X-RAY
Patient Information:
Name: {name}
Birth Date: {dob}
Medical Record: {mrn}
Social Security: {ssn}
Study Date: {scan_date}
Institution: {hospital_name}
Location: {hospital_location}
Attending Physician: Dr. {doctor}
Examination Type: {scan_type.title()}
Region(s) Examined: {laterality.title()}
Interpretation: {findings.capitalize()}
Impression: {impression}
Recommended Follow-up: {follow_up}
Patient Address: {address}
Hometown: {patient_location}
Phone: {phone}
Email: {email}
Additional Identifiers: Vehicle ID {vehicle_id}, License Plate {license_plate}
        """,

        # Template 8: Hospital system format
        f"""
{hospital_name} - RADIOLOGY SERVICES
{hospital_location}
PATIENT: {name} (DOB: {dob})
EXAM DATE: {scan_date}
RADIOLOGIST: Dr. {doctor}
STUDY TYPE: {scan_type.title()}
LATERALITY: {laterality.title()}
MEDICAL RECORD: {mrn}
PATIENT ACCOUNT: {account_number}
FINDINGS: {findings.capitalize()}
ASSESSMENT: {impression}
RECOMMENDATIONS: {follow_up}
PATIENT DEMOGRAPHICS:
Current Address: {address}
Place of Origin: {patient_location}
Telephone: {phone}
Email Address: {email}
Additional IDs: Certificate #{certificate_number}, Study #{study_id}
        """,

        # Template 9: Quick reference format
        f"""
X-RAY RESULTS - {scan_date}
Pt: {name} (born {dob}) | MRN: {mrn} | ID: {patient_id}
Facility: {hospital_name}, {hospital_location}
Radiologist: Dr. {doctor}
Exam: {scan_type.title()} ({laterality})
Results: {findings.capitalize()}
Impression: {impression}
Action: {follow_up}
Address: {address} (from {patient_location})
Contact: {phone} / {email}
Ref: Health Plan {health_plan_id}, License {license_number}
        """,

        # Template 10: Formal medical report
        f"""
DEPARTMENT OF RADIOLOGY
{hospital_name}
{hospital_location}

X-RAY REPORT

PATIENT DEMOGRAPHICS:
Full Name: {name}
Date of Birth: {dob}
Medical Record Number: {mrn}
Social Security Number: {ssn}
Current Address: {address}
City of Origin: {patient_location}
Contact Phone: {phone}
Email: {email}

EXAMINATION DETAILS:
Date of Examination: {scan_date}
Procedure: {scan_type.title()}
Region(s) Examined: {laterality.title()}
Interpreting Physician: Dr. {doctor}

FINDINGS AND IMPRESSION:
Imaging demonstrates {findings}. No acute pathology identified.
Impression: {impression}
Clinical Recommendation: {follow_up}

PATIENT IDENTIFIERS:
Account Number: {account_number}
Case Number: {case_number}
Device Serial: {device_serial}
        """
    ]

    template = random.choice(templates).strip()

    # Annotate PHI spans with the 6 entity categories
    entities = []
    for ent_text, label in [
        (name, "NAME"),
        (doctor, "NAME"),
        (dob, "DATE"),
        (scan_date, "DATE"),
        (address, "LOCATION"),
        (hospital_location, "LOCATION"),
        (patient_location, "LOCATION"),
        (hospital_name, "HOSPITAL"),
        (mrn, "ID"),
        (ssn, "ID"),
        (health_plan_id, "ID"),
        (account_number, "ID"),
        (license_number, "ID"),
        (certificate_number, "ID"),
        (license_plate, "ID"),
        (vehicle_id, "ID"),
        (device_serial, "ID"),
        (patient_id, "ID"),
        (case_number, "ID"),
        (study_id, "ID"),
        (phone, "CONTACT"),
        (email, "CONTACT"),
    ]:
        start = template.find(ent_text)
        if start != -1:
            end = start + len(ent_text)
            entities.append({"start": start, "end": end, "label": label, "text": ent_text})

    return {"text": template, "entities": entities}



def generate_mri_report():
    hospitals = [
        ("City General Hospital", "123 Health St, Cityville, State"),
        ("Westside Medical Center", "456 Care Ave, Townsville, State"),
        ("Eastside Radiology Clinic", "789 Wellness Rd, Villagetown, State"),
        ("Central Imaging Institute", "101 Recovery Blvd, Metrocity, State"),
        ("Regional Diagnostic Center", "555 Healing Ln, Riverside, State"),
        ("Metropolitan Radiology Unit", "777 Therapy St, Centraltown, State"),
        ("Community Imaging Services", "999 Clinic Dr, Southside, State"),
        ("Advanced Radiology Institute", "321 Treatment Ave, Northville, State"),
    ]

    patient_locations = [
        "Cityville", "Townsville", "Villagetown", "Metrocity",
        "Lakeside", "Hilltown", "Riverside", "Centraltown",
        "Southside", "Northville", "Westbrook", "Eastwood"
    ]

    hospital_name, hospital_location = random.choice(hospitals)
    patient_location = random.choice(patient_locations)
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%B %d, %Y")
    address = fake.address().replace("\n", ", ")
    doctor = fake.name()
    phone = fake.phone_number()
    email = fake.email()
    mrn = fake.bothify(text="#####")
    ssn = fake.ssn()
    health_plan_id = fake.bothify(text="########")
    account_number = fake.bothify(text="#######")
    license_number = fake.bothify(text="#######")
    certificate_number = fake.bothify(text="#####")
    license_plate = fake.bothify(text="???###")
    vehicle_id = fake.bothify(text="########")
    device_serial = fake.bothify(text="DEV-SN-######")
    patient_id = fake.bothify(text="PT-####")
    case_number = fake.bothify(text="CN-######")
    study_id = fake.bothify(text="STU-#####")
    scan_date = fake.date()
    scan_type = random.choice(["MRI RIGHT ELBOW", "MRI LEFT SHOULDER", "MRI RIGHT KNEE", "MRI LEFT ANKLE", "MRI LUMBAR SPINE"])
    laterality = random.choice(["left", "right", "bilateral"])

    clinical_info = random.choice([
        "Evaluate persistent right elbow pain. Status post fall.",
        "Work related injury on September 21, 2015, assess for traumatic tear left rotator cuff with superior shoulder pain and weakness.",
        "Assess chronic knee pain and possible meniscal tear.",
        "Evaluate left ankle pain and swelling following sports injury.",
        "Assess lower back pain and possible disc herniation."
    ])

    technique = random.choice([
        "Axial T2 axial T1 and coronal T1 axial T1 coronal STIR coronal gradient echo sagittal STIR",
        "Axial PD FS, coronal PD FS and PD, sagittal T1 and PD FS imaging is performed through the left shoulder without contrast.",
        "Sagittal T1, axial T2, coronal STIR, and axial PD FS imaging performed through the right knee.",
        "Axial T1, coronal T2, sagittal PD FS, and axial STIR imaging performed through the left ankle.",
        "Sagittal T1, axial T2, coronal STIR, and axial PD FS imaging performed through the lumbar spine."
    ])

    findings = random.choice([
        "There is a moderate elbow effusion. There is no osteochondral defect of the capitellum. There is no intraarticular loose fragment or body.",
        "There is mild distal insertional tendinosis with minimal articular sided fraying of the distal aspect of the supraspinatus tendon. There is a 3 mm low-grade longitudinal interstitial tear involving the supraspinatus tendon at the distal attachment site.",
        "There is evidence of a meniscal tear in the medial meniscus. There is mild joint effusion and synovitis.",
        "There is a fracture of the lateral malleolus with associated soft tissue edema. There is no evidence of ligamentous injury.",
        "There is evidence of disc herniation at L4-L5 with associated nerve root compression. There is mild spinal stenosis."
    ])

    impression = random.choice([
        "Moderate elbow effusion. Evidence of a fracture of the radial head. Low-grade ligamentous sprains of the medial collateral ligament and lateral ulnar collateral ligament proximally. Mild tendinosis of the common extensor tendon origin.",
        "Mild supraspinatus tendinosis with minimal articular sided fraying of the distal tendon and a 3 mm low grade interstitial tear at the distal attachment site. Type II SLAP lesion and mild tendinosis of the intra-articular portion of the long head biceps tendon.",
        "Meniscal tear in the medial meniscus. Mild joint effusion and synovitis.",
        "Fracture of the lateral malleolus with associated soft tissue edema. No evidence of ligamentous injury.",
        "Disc herniation at L4-L5 with associated nerve root compression. Mild spinal stenosis."
    ])

    follow_up = random.choice([
        "Annual follow-up recommended.",
        "Ultrasound follow-up in 6 months.",
        "Biopsy recommended for further evaluation.",
        "Short interval follow-up advised.",
        "Continue routine monitoring.",
        "Diagnostic imaging in 12 months.",
        "Immediate surgical consultation required.",
        "MRI correlation suggested."
    ])

    templates = [
        # Template 1: Standard clinical format
        f"""
MRI REPORT
Patient: {name}
DOB: {dob}
MRN: {mrn}
Study Date: {scan_date}
Facility: {hospital_name}, {hospital_location}
Interpreting Physician: Dr. {doctor}

CLINICAL INFORMATION: {clinical_info}

TECHNIQUE: {technique}

FINDINGS: {findings}

IMPRESSION: {impression}
RECOMMENDATION: {follow_up}

Patient contact information: {phone}, {email}
Residence: {address}
Originally from: {patient_location}
Account #: {account_number}
        """,

        # Template 2: Narrative format
        f"""
On {scan_date}, patient {name} (born {dob}) presented to {hospital_name} located at {hospital_location} for {scan_type}.
The examination was performed and interpreted by Dr. {doctor}. Imaging revealed {findings}.
Assessment: {impression}.
Clinical recommendation: {follow_up}
Patient identifiers: MRN {mrn}, SSN {ssn}, Study ID {study_id}.
Contact: {phone} / {email}. Current address: {address}, hometown: {patient_location}.
        """,

        # Template 3: Structured clinical note
        f"""
RADIOLOGY REPORT - MRI
Date of Service: {scan_date}
Patient Name: {name}
Date of Birth: {dob}
Medical Record Number: {mrn}
Patient ID: {patient_id}
Facility: {hospital_name}
Address: {hospital_location}
Radiologist: Dr. {doctor}
Procedure: {scan_type.title()}
Laterality: {laterality.title()}
Results: {findings.capitalize()}
Impression: {impression}
Follow-up: {follow_up}
Patient Demographics: Resides at {address}, originally from {patient_location}
Emergency Contact: {phone}, {email}
Case Number: {case_number}
        """,

        # Template 4: Brief clinical summary
        f"""
{name} (DOB: {dob}, MRN: {mrn}) underwent {scan_type} on {scan_date} at {hospital_name}, {hospital_location}.
Dr. {doctor} reviewed the {laterality} region images showing {findings}.
Classification: {impression}. Plan: {follow_up}
Demographics: {address}, from {patient_location}
Contact: {phone}, {email}
Reference: Account #{account_number}, Certificate #{certificate_number}
        """,

        # Template 5: Detailed assessment format
        f"""
MRI EXAMINATION REPORT
Examination performed on {scan_date} for {name} (date of birth {dob})
Location: {hospital_name} - {hospital_location}
Interpreting radiologist: Dr. {doctor}
Medical identifiers: MRN {mrn}, Health Plan ID {health_plan_id}, License #{license_number}
Procedure: {scan_type.title()} examination of {laterality} region
Clinical findings: {findings.capitalize()}
Impression: {impression}
Management recommendation: {follow_up}
Patient residence: {address}
Originally from: {patient_location}
Preferred contact: {phone} or {email}
        """,

        # Template 6: Abbreviated report style
        f"""
{hospital_name} ({hospital_location}) - MRI RESULTS
Patient: {name}, DOB {dob}
Date: {scan_date}
Physician: Dr. {doctor}
Type: {scan_type.title()} - {laterality} region
Findings: {findings.capitalize()}
Impression: {impression}
Next steps: {follow_up}
Patient info: {address}, originally {patient_location}
Contact: {phone}, {email}
IDs: MRN {mrn}, Account {account_number}, Device {device_serial}
        """,

        # Template 7: Comprehensive clinical documentation
        f"""
RADIOLOGY REPORT - MRI
Patient Information:
Name: {name}
Birth Date: {dob}
Medical Record: {mrn}
Social Security: {ssn}
Study Date: {scan_date}
Institution: {hospital_name}
Location: {hospital_location}
Attending Physician: Dr. {doctor}
Examination Type: {scan_type.title()}
Region(s) Examined: {laterality.title()}
Interpretation: {findings.capitalize()}
Impression: {impression}
Recommended Follow-up: {follow_up}
Patient Address: {address}
Hometown: {patient_location}
Phone: {phone}
Email: {email}
Additional Identifiers: Vehicle ID {vehicle_id}, License Plate {license_plate}
        """,

        # Template 8: Hospital system format
        f"""
{hospital_name} - RADIOLOGY SERVICES
{hospital_location}
PATIENT: {name} (DOB: {dob})
EXAM DATE: {scan_date}
RADIOLOGIST: Dr. {doctor}
STUDY TYPE: {scan_type.title()}
LATERALITY: {laterality.title()}
MEDICAL RECORD: {mrn}
PATIENT ACCOUNT: {account_number}
FINDINGS: {findings.capitalize()}
ASSESSMENT: {impression}
RECOMMENDATIONS: {follow_up}
PATIENT DEMOGRAPHICS:
Current Address: {address}
Place of Origin: {patient_location}
Telephone: {phone}
Email Address: {email}
Additional IDs: Certificate #{certificate_number}, Study #{study_id}
        """,

        # Template 9: Quick reference format
        f"""
MRI RESULTS - {scan_date}
Pt: {name} (born {dob}) | MRN: {mrn} | ID: {patient_id}
Facility: {hospital_name}, {hospital_location}
Radiologist: Dr. {doctor}
Exam: {scan_type.title()} ({laterality})
Results: {findings.capitalize()}
Impression: {impression}
Action: {follow_up}
Address: {address} (from {patient_location})
Contact: {phone} / {email}
Ref: Health Plan {health_plan_id}, License {license_number}
        """,

        # Template 10: Formal medical report
        f"""
DEPARTMENT OF RADIOLOGY
{hospital_name}
{hospital_location}

MRI REPORT

PATIENT DEMOGRAPHICS:
Full Name: {name}
Date of Birth: {dob}
Medical Record Number: {mrn}
Social Security Number: {ssn}
Current Address: {address}
City of Origin: {patient_location}
Contact Phone: {phone}
Email: {email}

EXAMINATION DETAILS:
Date of Examination: {scan_date}
Procedure: {scan_type.title()}
Region(s) Examined: {laterality.title()}
Interpreting Physician: Dr. {doctor}

FINDINGS AND IMPRESSION:
Imaging demonstrates {findings}. No acute pathology identified.
Impression: {impression}
Clinical Recommendation: {follow_up}

PATIENT IDENTIFIERS:
Account Number: {account_number}
Case Number: {case_number}
Device Serial: {device_serial}
        """
    ]

    template = random.choice(templates).strip()

    # Annotate PHI spans with the 6 entity categories
    entities = []
    for ent_text, label in [
        (name, "NAME"),
        (doctor, "NAME"),
        (dob, "DATE"),
        (scan_date, "DATE"),
        (address, "LOCATION"),
        (hospital_location, "LOCATION"),
        (patient_location, "LOCATION"),
        (hospital_name, "HOSPITAL"),
        (mrn, "ID"),
        (ssn, "ID"),
        (health_plan_id, "ID"),
        (account_number, "ID"),
        (license_number, "ID"),
        (certificate_number, "ID"),
        (license_plate, "ID"),
        (vehicle_id, "ID"),
        (device_serial, "ID"),
        (patient_id, "ID"),
        (case_number, "ID"),
        (study_id, "ID"),
        (phone, "CONTACT"),
        (email, "CONTACT"),
    ]:
        start = template.find(ent_text)
        if start != -1:
            end = start + len(ent_text)
            entities.append({"start": start, "end": end, "label": label, "text": ent_text})

    return {"text": template, "entities": entities}

def generate_surgery_report():
    procedures = [
        "laparoscopic cholecystectomy",
        "open appendectomy",
        "knee arthroscopy",
        "thyroidectomy"
    ]
    
    indications = [
        "symptomatic cholelithiasis",
        "acute suppurative appendicitis",
        "medial meniscus tear",
        "enlarging thyroid nodule with compressive symptoms"
    ]

    findings_list = [
        "a distended, inflamed gallbladder with multiple cholesterol stones",
        "a gangrenous appendix with surrounding purulent fluid",
        "a complex tear of the medial meniscus with joint effusion",
        "a multinodular thyroid with suspicious features under ultrasound"
    ]

    complications_list = [
        "Procedure was completed without intraoperative complications.",
        "Minimal bleeding encountered, managed effectively with bipolar cautery.",
        "Extensive adhesions encountered, requiring sharp dissection.",
        "Anatomical variations were present but navigated without incident."
    ]

    outcomes = [
        "Patient tolerated the procedure well and was transferred to the PACU in stable condition.",
        "Surgical goals were achieved with no deviation from planned protocol.",
        "Post-operative vitals were stable, and no immediate concerns were noted.",
        "Wound closed with subcuticular sutures; hemostasis confirmed prior to closure."
    ]

    hospitals = [
        ("Mount Atlas Surgical Centre", "321 Recovery Blvd, Healthtown, State"),
        ("Pineview General Hospital", "908 Wellness Way, Care City, State"),
        ("Surgical Arts Pavilion", "150 Operation Ave, MediCity, State"),
        ("North Ridge Medical Center", "45 Surgeon St, Vitalville, State")
    ]

    # Random Selections
    procedure = random.choice(procedures)
    indication = random.choice(indications)
    findings = random.choice(findings_list)
    complications = random.choice(complications_list)
    outcome = random.choice(outcomes)
    hospital_name, hospital_address = random.choice(hospitals)

    # Patient & Doctor Info
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=85).strftime("%B %d, %Y")
    doctor = fake.name()
    phone = fake.phone_number()
    email = fake.email()
    address = fake.address().replace("\n", ", ")
    mrn = fake.bothify(text="MRN-#####")
    surgery_date = fake.date()

    templates = [
        # Narrative operative summary
        f"""
        On {surgery_date}, patient {name} (DOB: {dob}) underwent a {procedure} at {hospital_name} ({hospital_address}) for {indication}. 
        Intraoperatively, {findings} were noted by the attending surgeon, Dr. {doctor}. 
        {complications} {outcome}
        For follow-up or questions, please contact {phone} or email {email}. Patient address: {address}, MRN: {mrn}.
        """,

        # Structured case report
        f"""
        Operative Report
        -----------------
        Patient Name: {name}
        Date of Birth: {dob}
        MRN: {mrn}
        Date of Procedure: {surgery_date}
        Surgeon: Dr. {doctor}
        Facility: {hospital_name}, {hospital_address}

        Procedure: {procedure}
        Indication: {indication}
        Findings: {findings}
        Complications: {complications}
        Outcome: {outcome}

        Contact Info:
        Phone: {phone}
        Email: {email}
        Address: {address}
        """,

        # Clinical vignette format
        f"""
        Patient {name}, a {random.randint(25, 78)}-year-old individual, presented with {indication} and was taken to the OR for a {procedure} on {surgery_date}. 
        The surgical team at {hospital_name} led by Dr. {doctor} identified {findings} during the procedure. 
        {complications} {outcome} The patient resides at {address}. Contact via phone: {phone}, or email: {email}. MRN: {mrn}.
        """,

        # Academic conference summary
        f"""
        Case Summary:
        - Patient: {name} ({dob})
        - Procedure: {procedure}
        - Indication: {indication}
        - Findings: {findings}
        - Complications: {complications}
        - Outcome: {outcome}
        - Surgeon: Dr. {doctor}
        - Date: {surgery_date}
        - Facility: {hospital_name}, {hospital_address}

        For further discussion or data access, contact {phone} | {email}. MRN: {mrn}, Address: {address}.
        """,

        # Discharge note excerpt
        f"""
        Discharge Note:
        {name} (DOB {dob}) underwent a successful {procedure} on {surgery_date} at {hospital_name}, indicated for {indication}. 
        Intraoperative findings included: {findings}. {complications} {outcome}
        Discharge instructions provided. Contact: {phone} | {email}, MRN: {mrn}, Address: {address}.
        """
    ]

    template = random.choice(templates).strip()

    # Annotate PHI spans
    entities = []
    for ent_text, label in [
        (name, "NAME"),
        (dob, "DATE"),
        (doctor, "NAME"),
        (address, "LOCATION"),
        (hospital_name, "HOSPITAL"),
        (hospital_address, "LOCATION"),
        (mrn, "ID"),
        (phone, "CONTACT"),
        (email, "CONTACT"),
    ]:
        start = template.find(ent_text)
        if start != -1:
            end = start + len(ent_text)
            entities.append({"start": start, "end": end, "label": label, "text": ent_text})

    return {"text": template, "entities": entities}



def generate_lab_report():
    test_types = [
        "complete blood count (CBC)",
        "basic metabolic panel (BMP)",
        "lipid panel",
        "coagulation profile",
        "liver function test (LFT)",
        "thyroid-stimulating hormone (TSH) panel"
    ]

    indications = [
        "routine annual health assessment",
        "persistent fatigue and dizziness",
        "renal function monitoring in chronic kidney disease",
        "assessment of lipid levels due to family history of cardiovascular disease",
        "evaluation of liver enzymes due to suspected hepatitis",
        "thyroid dysfunction symptoms including weight gain and cold intolerance"
    ]

    findings_list = [
        "values within expected physiological range",
        "microcytic anemia with decreased hemoglobin and hematocrit",
        "elevated creatinine and reduced eGFR indicating Stage 2 CKD",
        "LDL cholesterol above optimal limits with low HDL",
        "elevated ALT and AST consistent with hepatocellular injury",
        "elevated TSH with low T3 and T4 suggesting primary hypothyroidism"
    ]

    impressions = [
        "No clinically significant abnormalities identified.",
        "Anemia consistent with iron deficiency; further evaluation recommended.",
        "Renal insufficiency suspected—nephrology referral suggested.",
        "Hyperlipidemia noted; dietary modification and statin therapy advised.",
        "Liver function derangement suggestive of mild hepatic inflammation.",
        "Thyroid profile supports diagnosis of hypothyroidism; endocrinology consult may be beneficial."
    ]

    # Random selections
    test_type = random.choice(test_types)
    indication = random.choice(indications)
    findings = random.choice(findings_list)
    impression = random.choice(impressions)

    # Patient & Doctor Info
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=85).strftime("%B %d, %Y")
    doctor = fake.name()
    phone = fake.phone_number()
    email = fake.email()
    address = fake.address().replace("\n", ", ")
    mrn = fake.bothify(text="MRN-#####")
    scan_date = fake.date()

    templates = [
        # Clinical summary style
        f"""
        Patient {name} (DOB: {dob}) underwent a {test_type} on {scan_date} due to {indication}. 
        Laboratory results were interpreted by Dr. {doctor} and revealed the following: {findings}. 
        Impression: {impression}. For inquiries, reach out at {phone} or email {email}. Address: {address}. MRN: {mrn}.
        """,

        # Progress note style
        f"""
        Date: {scan_date}
        Patient: {name} (DOB: {dob}) | MRN: {mrn}
        Test Performed: {test_type}
        Indication: {indication}
        Findings: {findings}
        Impression: {impression}
        Ordering Physician: Dr. {doctor}
        Contact: {phone}, {email}
        Residence: {address}
        """,

        # Research-style narrative
        f"""
        As part of routine diagnostics on {scan_date}, {name} (born {dob}) had a {test_type} performed to investigate {indication}. 
        Results revealed {findings}. The impression provided by Dr. {doctor} indicated: {impression}. 
        For medical records or inquiries, the patient can be contacted via phone ({phone}) or email ({email}). Home address: {address}. MRN: {mrn}.
        """,

        # Academic abstract style
        f"""
        Case: {name} (DOB: {dob}), MRN: {mrn}
        Test: {test_type}
        Indication: {indication}
        Findings: {findings}
        Impression: {impression}
        Performed on: {scan_date} | Interpreting Clinician: Dr. {doctor}
        Contact: {phone} | {email}
        Address: {address}
        """,

        # Discharge lab summary style
        f"""
        Lab Summary: On {scan_date}, {name} (DOB {dob}) underwent a {test_type} in context of {indication}. 
        Notable results: {findings}. Interpreting physician, Dr. {doctor}, noted: {impression}. 
        MRN: {mrn}. Contact details include {phone} and {email}. Patient resides at {address}.
        """
    ]

    template = random.choice(templates).strip()

    # Annotate PHI spans
    entities = []
    for ent_text, label in [
        (name, "NAME"),
        (dob, "DATE"),
        (doctor, "NAME"),
        (address, "LOCATION"),
        (mrn, "ID"),
        (phone, "CONTACT"),
        (email, "CONTACT"),
    ]:
        start = template.find(ent_text)
        if start != -1:
            end = start + len(ent_text)
            entities.append({"start": start, "end": end, "label": label, "text": ent_text})

    return {"text": template, "entities": entities}

def reports():
    report_generators = {
    "Lab Report": lambda: generate_lab_report(),
    "MRI Report": lambda: generate_mri_report(),
    "X-ray Report": lambda: generate_xray_report(),
    "Mammography Report": lambda: generate_mammography_report(),
    "Summary Report": lambda: generate_discharge_summary(),
    "CT Scan Report": lambda: generate_ct_scan_report(),
    "Surgery Report": lambda: generate_surgery_report(),
}

    rows = []
    for report_type, generator in report_generators.items():
        reports = [generator() for _ in range(250)]
        for report in reports:
            rows.append({
                "report_type": report_type,
                "text": report["text"],
                "entities": report["entities"]
            })

    df = pd.DataFrame(rows)
    return df

def main():
    df = reports()
    # Save the DataFrame to a CSV file
    df.to_csv("medical_reports.csv", index=False)
    print("Medical reports generated and saved to 'medical_reports.csv'.")

if __name__ == "__main__":
    # Generate a sample report
    main()