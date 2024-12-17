from langchain_groq import ChatGroq
from utils.ScrapeClass import ScrapeJobDetails
from utils.LLMResponse import GenerateLLMResponse
import pandas as pd


def initialize_():
    api = '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'
    llm = ChatGroq(groq_api_key=api,
                   model="llama-3.3-70b-versatile", temperature=0.3)

    resume = '''

        Skills Summary
        • Expertise: RAGs, AI Systems Design, End-to-End AI Deployment, LLM Integration & Fine-tuning ,Cross Platform App
        Development, Web Development, Applied Deep Learning
        • Frameworks: React Native, React.js, Next.js, FastAPI, Selenium, Streamlit
        • Languages: Python, Java, Kotlin, Swift, C++, JavaScript, TypeScript
        • Tools: Git, Docker, Firebase, CVAT, Tailwind CSS, Supabase, Neo4j
        • Libraries: Tensorflow, Langchain, NLTK, OpenCV, Huggingface, Llamaindex, Transformers
        • Soft Skills: Proactive Planning, Team Leadership, Adaptability, Cross-Functional Collaboration
        Education
        •
        Bennett University - The Times Group Greater Noida, India
        Bachelor of Technology - Computer Science and Engineering; GPA: 8.9 (2024) Sep 2022 - June 2026
        Specialization: Artificial Intelligence
        Experience
        •
        Resilient Moment Maryland, United States · Remote
        AI Intern On-Going
        ◦ Reduced the API overhead by 95% by developing an automation script for translation.
        ◦ Developed a Native IOS libary for text-to-speech (TTS)
        ◦ Utilized Technologies: Python, IOS, XCode, React Native
        •
        Samsung Display Noida, India · Hybrid
        Computer Vision Engineer (Consultant) Apr. 2024 – Sept. 2024
        ◦ Saved 40,000 annual work hours for the world’s largest smartphone display manufacturing plant by automating the
        calibration of manufacturing robots up to 20x faster.
        ◦ Achieved length measurement precision up to 1/10th of a millimeter with a Computer Vision algorithm.
        ◦ Optimized models for edge deployment, achieving a mean inference time of 300ms and a model size of 10MB.
        ◦ Utilized Technologies: OpenCV, TensorFlow, TFLite, CVAT, Android, Java, Python
        •
        Mobilon Greater Noida, India
        Founder & Technical Lead Aug. 2023 - May 2024
        ◦ Co-founded and lead the technical team of 40+ members
        ◦ Supervised the development of multiple university-sponsored projects and deployed 3 apps.
        •
        Food Future Foundation New Delhi, India
        Full Stack Intern Jan. 2024 - March 2024
        ◦ Developed a platform for a non-profit organization (NPO) to connect education experts and gather their reviews on a
        curriculum designed to spread food literacy in India.
        ◦ Utilized Technologies: NextJS, NextUI, Supabase
        Projects
        ◦ ClinGraph: A medical diagnosis tool with RAG architecture that outperforms LLM fine-tuning. Tech: Python, FastAPI,
        Next.js, Neo4j, Docker, Langchain (October 2024)
        ◦ Paligemma VLM: Implementation of Paligemma vision language model from scratch in pytorch Tech: Python, Pytorch
        (Sept. 2024)
        ◦ Federated Learning: Employed a federated learning environment for training Resnet50 & Vgg16 by utilizing client’s
        system while maintaining user-data privacy. Tech: Python, FastAPI, Tensorflow (April 2024)
        ◦ Rag-App: Developed a Retrieval-Augmented Generation (RAG) based platform enabling users to query with any uploaded
        document. Tech: Python, Llama Index, FastAPI, NextJS (March 2024)
        ◦ Not-by-Bot: Built a marketplace for authentic AI-free content, verified using an LSTM model trained on 12k essays with
        high accuracy. Tech: Python, NextJS, FastAPI, Prismic CMS, Supabase (March 2024)
        ◦ Culinary Compass: Developed a go-to recipe app offering pantry-friendly, global, & personalized meal recommendations
        tailored to nutritional goals. Tech: React Native, Firebase, Tailwind CSS (November 2023)
        Achievements
        ◦ Winner, Industry Hackathon: Nom-Nom: Cut calories, not taste. App to plan meals and focus on health. Solved
        challenge for an actual company.
        ◦ 3rd, IIIT-Delhi Fork-It Hackathon: Culinary Compass: A kitchen companion developed for providing recipe
        recommendations along with their instructions.
        ◦ 7th, Internal Smart India Hackathon: Web Connect: A blockchain based secure video conferencing platform
        ◦ Courses/Certificates: Deep Learning Specialization, Machine Learning Specialization, Meta Android Developer


        '''

    cover_letter = '''

        For me, programming has always been about building solutions that can actually make an impact on people’s day
        to day life. That’s exactly why feels like the right place for me to progress not just as an engineer, but as
        someone who can work in a diverse team working together to deliver real-world products. The idea of seeing a
        product I’ve worked on being used in the real world, and actually being impactful, really excites me and has always
        driven me.
        Throughout my career I have collaborated with a diverse team of developers to tackle real world challenges. I am
        an intern at Resilient Moment, where I am working on a cross-platform app designed for stress management by
        employing real-word therapies. In summers this year, I implemented a pixel-precise computer vision algorithm
        with on-device AI deployment for Samsung Display which increased their efficiency by 20 times. I am currently
        working on building efficient RAG pipelines along with understanding LLM integration to refine user experience and
        fine-tuning it for specific tasks. With these diverse experiences I have learnt to focus on designing AI solutions that
        prioritize real-world applicability, ensuring end-to-end solution from development to deployment stage.
        Last year, I co-founded the Mobile App Development Club at our university, where I led a team of 40+ enthusiasts
        to develop university-sponsored projects. I supervised the entire lifecycle of app development, from conducting
        research and incorporating user feedback to final deployment. We conducted surveys with artists and small creators
        from our university, tailoring the app specifically to their requirements. Our apps gained significant attention from
        our university’s incubation center, with one of the apps, ’Beeuzine,’ eventually becoming a startup. I now serve as a
        mentor to the team.
        My hands-on experience in this domain, addressing various challenges, has refined my ability to design user-focused
        products. My enthusiasm for using the modern day technology to craft solutions matches the vision of ,
        making me a perfect fit for this role. I look forward to the chance to discuss how my skills and experiences can align
        with the team’s goals to drive impactful solutions

        '''

    instructions = '''
    0. **Adopt a conversational yet professional tone**: Write naturally as if you're explaining your interest and value to a hiring manager, avoiding overly formal or robotic language.

    1. **Start with a specific reference to the company**: Mention a project, product, or initiative that directly aligns with your skills or experience. Research the company's AI-related work and integrate it naturally into the response. Avoid generic statements like "I'm excited about your innovation."

    2. **Showcase your unique abilities and proven expertise**: Highlight your strongest abilities, previous experience, and measurable achievements that make you uniquely qualified for this role. Draw examples directly from your past work, such as specific tools (e.g., LLMs, RAG architectures) or accomplishments (e.g., "improved response accuracy by 25%").

    3. **Demonstrate clear value to the company**: Link your achievements and skills to the company’s goals or challenges. Instead of saying "I can make an impact," explain exactly how your expertise will solve their problems, improve processes, or drive innovation.

    4. **Provide measurable results with context**: Use one strong example with metrics to back up your claims (e.g., "I implemented a this at that that improved performance by this percent "). Avoid inventing metrics or offering vague claims.

    5. **Make every sentence count**: Remove repetition or filler. Each sentence should serve a clear purpose—whether it's showing value, demonstrating expertise, or connecting to the company's mission.

    6. **End with a confident, specific conclusion**: Reinforce how your unique skills will help the company succeed. Replace weak statements like "I’m excited to help" with clear, tailored contributions (e.g., "I’m eager to optimize your AI pipeline and drive measurable performance gains for your team").

    7. **Keep it concise and focused**: Limit the response to **100-150 words** while retaining clarity, specificity, and impact. Be enthusiastic but to the point.

    8. Replace metrics & results that are not mentioned in my resume & cover letter. Don't invent results. Only use values that are in resume & cover letter
    '''

    task = "Question: What interests you about working for this company?"

    return resume, cover_letter, instructions, task, llm


def loop(list_: list, resume: str, cover_letter: str, task: str, instructions: str, llm):
    object_ = ScrapeJobDetails(list_)

    response_dict = object_._run()

    lr = GenerateLLMResponse(llm)

    resultList = []

    for url in list_:

        context = lr._extract_relevant_details(
            resume=resume, job_desc=response_dict[url]
        )

        custom_prompt = lr._buildPrompt(
            context = context, cover_letter=cover_letter, jobData=response_dict[url], task=task)

        final_response = lr._runInferenceLoop(
            instruction=instructions, prompt=custom_prompt)

        resultList.append(final_response)
        print(final_response)

    return resultList


if __name__ == "__main__":

    list_ = ['https://wellfound.com/jobs/3150781-ai-engineering-intern']

    resume, cover_letter, instructions, task, llm = initialize_()

    result_list = loop(list_, resume=resume, cover_letter=cover_letter, task=task, instructions=instructions, llm=llm)

    df = pd.DataFrame({
        'Links': list_,
        'Responses': result_list
    })

    df.to_csv('final.csv')
