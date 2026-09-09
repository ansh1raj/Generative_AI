from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="ddy0126/tinylama-Fine-tuned",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat_model = ChatHuggingFace(llm=llm)
response = chat_model.invoke("what is deep learning?")
print(response.content)

# pytorch has no wheel for current python version 3.14. so, just install pytorch and run model would install locally in your system. 
  