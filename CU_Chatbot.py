!pip install transformers sentence-transformers torch nltk

import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

faq_data = {
    "faqs": [
        {"question": "What courses are offered at Chandigarh University?", "answer": "Chandigarh University offers undergraduate, postgraduate, and doctoral programs in various fields including engineering, management, law, architecture, media, and design."},
        {"question": "What is the admission process at Chandigarh University?", "answer": "Admissions are based on the CUCET entrance exam, followed by a personal interview or group discussion for some programs."},
        {"question": "What are the hostel facilities like at Chandigarh University?", "answer": "The university provides hostel accommodations with modern amenities, including Wi-Fi, medical services, gym, and mess facilities."},
        {"question": "What are the placement statistics for engineering at Chandigarh University?", "answer": "The placement rate for engineering students is 90%, with top recruiters like Microsoft, Amazon, and Google. The highest package offered was INR 52 LPA."},
        {"question": "Does Chandigarh University offer scholarships?", "answer": "Yes, Chandigarh University offers scholarships based on CUCET scores, board results, and other criteria. Scholarships can cover up to 100% of tuition fees."},
        {"question": "What are the campus timings?", "answer": "Chandigarh University campus operates from 9 AM to 5 PM for most courses, although labs and libraries are open until 9 PM."},
        {"question": "How many students are enrolled at Chandigarh University?", "answer": "As of 2023, over 30,000 students are enrolled at Chandigarh University across various programs."},
        {"question": "What is the fee structure for the B.Tech program?", "answer": "The annual tuition fee for B.Tech at Chandigarh University is INR 1.8 lakhs. Additional costs for hostel and mess are around INR 80,000 per year."},
        {"question": "What is the highest package offered for MBA graduates?", "answer": "In 2023, the highest package for MBA graduates was INR 28 LPA, offered by a global consulting firm."},
        {"question": "Which companies visit Chandigarh University for placements?", "answer": "Top companies like Microsoft, Google, Amazon, IBM, Infosys, Deloitte, and Wipro visit Chandigarh University for campus placements."},
        {"question": "What extracurricular activities are available?", "answer": "Chandigarh University offers sports, cultural fests, student clubs (coding, arts, etc.), and volunteer activities for overall development."},
        {"question": "What is the application process for international students?", "answer": "International students can apply online through the Chandigarh University website. They must submit academic transcripts and proof of English proficiency (IELTS/TOEFL)."},
        {"question": "How do I pay the application fee for CUCET?", "answer": "You can pay the CUCET application fee online via credit card, debit card, or net banking. The fee is INR 1,000."},
        {"question": "What is the ranking of Chandigarh University?", "answer": "Chandigarh University is ranked #29 among Indian universities by the NIRF 2023 and holds a QS Asia ranking in the top 200."},
        {"question": "Is Chandigarh University accredited?", "answer": "Yes, Chandigarh University is NAAC A+ accredited and is recognized by UGC and AICTE."},
        {"question": "What is the eligibility for the PhD program?", "answer": "For PhD admissions, candidates must have a master’s degree in the relevant field with a minimum of 55% marks. Admission is based on a written test and interview."},
        {"question": "What is the deadline for CUCET registration?", "answer": "The deadline for CUCET 2024 Phase I registration is June 30, 2024."},
        {"question": "Are internships mandatory for all courses?", "answer": "Yes, internships are mandatory for most programs at Chandigarh University, including B.Tech, MBA, and Journalism."},
        {"question": "What is the duration of the B.Sc Nursing program?", "answer": "The B.Sc Nursing program at Chandigarh University is a 4-year undergraduate course."},
        {"question": "What are the research opportunities in biotechnology?", "answer": "Biotechnology students can participate in research projects in collaboration with industries, research institutes, and international universities."},
        {"question": "What is the highest package offered for law students?", "answer": "The highest package offered to law students in 2023 was INR 12 LPA."},
        {"question": "What languages are taught in the School of Languages?", "answer": "Chandigarh University offers courses in English, French, German, and Spanish."},
        {"question": "What sports facilities are available?", "answer": "The university provides facilities for cricket, basketball, football, tennis, badminton, and a gymnasium."},
        {"question": "How do I apply for hostel accommodation?", "answer": "Students can apply for hostel accommodation during admission by filling out the form on the university's website. Hostels are allocated on a first-come, first-served basis."},
        {"question": "What is the eligibility for the MBA program?", "answer": "For the MBA program, candidates must have a bachelor's degree with at least 50% marks. Admission is based on CUCET scores and GD/PI rounds."},
        {"question": "Are there student exchange programs?", "answer": "Yes, Chandigarh University has partnerships with over 200 universities worldwide, offering student exchange programs."},
        {"question": "What is the student-faculty ratio at Chandigarh University?", "answer": "The student-faculty ratio is 14:1, ensuring personalized attention for every student."},
        {"question": "What are the entrance exams accepted for the B.Arch program?", "answer": "Chandigarh University accepts NATA and JEE Main (Paper 2) scores for admission to the B.Arch program."},
        {"question": "How can I contact the admission office?", "answer": "You can contact the admission office via phone at +91-160-3051003 or email at admissions@cumail.in."},
        {"question": "Does Chandigarh University provide distance education?", "answer": "Yes, the university offers distance education programs in management, IT, and commerce through its distance learning center."},
        {"question": "How many research papers are published annually?", "answer": "Over 1,500 research papers were published by faculty and students of Chandigarh University in 2023."},
        {"question": "What are the placement opportunities for computer science students?", "answer": "Computer science students have excellent placement opportunities with companies like Google, Microsoft, Infosys, and Amazon, with an average package of INR 7 LPA."},
        {"question": "What medical facilities are available on campus?", "answer": "The university has a fully-equipped health center with doctors on-call 24/7, along with an ambulance service."},
        {"question": "What is the process for getting a scholarship?", "answer": "Scholarships are based on performance in the CUCET exam, with up to 100% fee waivers for top performers. There are also scholarships for sports and merit-based achievements."},
        {"question": "What types of projects do civil engineering students work on?", "answer": "Civil engineering students work on projects like smart cities, sustainable building materials, and infrastructure development in collaboration with the industry."},
        {"question": "Are there programs for skill development?", "answer": "Yes, Chandigarh University has a dedicated skill development center offering certification courses in areas like AI, data science, and IoT."},
        {"question": "What is the fee structure for the M.Tech program?", "answer": "The annual tuition fee for the M.Tech program is INR 1.5 lakhs. Scholarships are available for top-performing students."},
        {"question": "Does Chandigarh University have an incubation center?", "answer": "Yes, Chandigarh University has an incubation center that supports startups and entrepreneurship with mentoring, funding, and industry collaborations."},
        {"question": "How do I check my CUCET results?", "answer": "You can check your CUCET results on the official Chandigarh University website by entering your registration number and password."},
        {"question": "Does the university offer hostel accommodation for international students?", "answer": "Yes, Chandigarh University provides separate hostel accommodations for international students, with modern amenities and a multicultural environment."},
        {"question": "What is the scope of research in computer science?", "answer": "Research in computer science focuses on AI, machine learning, data analytics, cybersecurity, and blockchain technology. Students can collaborate with industries for practical exposure."},
        {"question": "How many faculty members are there at Chandigarh University?", "answer": "Chandigarh University has over 1,800 faculty members across various departments, many of whom hold PhDs from reputed institutions."},
        {"question": "What is the process for applying for a PhD?", "answer": "To apply for a PhD, candidates need to pass the CU-PhD entrance test, followed by an interview with the research committee."},
        {"question": "How can I get a transcript from Chandigarh University?", "answer": "Students can request transcripts by filling out the form available on the student portal and paying the required fee."},
        {"question": "Is there Wi-Fi available on campus?", "answer": "Yes, the entire Chandigarh University campus is Wi-Fi enabled, with high-speed internet available in hostels, libraries, and academic blocks."},
        {"question": "What is the process to change my course?", "answer": "Students wishing to change their course must submit an application to the academic office, and approvals are granted based on seat availability and academic performance."},
        {"question": "Does the university offer dual degree programs?", "answer": "Yes, Chandigarh University offers dual degree programs such as BBA+MBA, B.Tech+M.Tech, and integrated law programs."}
    ]
}

with open('faq_chandigarh_university.json', 'w') as f:
    json.dump(faq_data, f, indent=4)

with open('faq_chandigarh_university.json') as f:
    faq_data = json.load(f)

faqs = faq_data['faqs']

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
faq_questions = [faq['question'] for faq in faqs]
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
qa_pipeline = pipeline("question-answering")

def find_closest_question(user_question):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(user_embedding, faq_embeddings)
    closest_idx = np.argmax(similarity_scores)
    return faqs[closest_idx]

def get_response(user_question):
    closest_faq = find_closest_question(user_question)
    context = closest_faq['answer']
    result = qa_pipeline(question=user_question, context=context)
    return result['answer'] if result['score'] > 0.1 else "Sorry, I don't have an answer for that."

def chatbot():
    print("Welcome to the Chandigarh University FAQ Chatbot! (type 'exit' to stop)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = get_response(user_input)
        print("Chatbot:", response)

chatbot()
