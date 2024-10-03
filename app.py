
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure that EOS token is present
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    model.resize_token_embeddings(len(tokenizer))

# define response genaration function
def generate_response(user_input, max_length = 100):
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    response_ids = model.generate(
        input_ids, 
        max_length=max_length,
        num_return_sequences =1,
        pad_token_id = tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    response = tokenizer.decode(response_ids[0], skip_special_tokens = True)
    return response

# Streamlit weeb app
st.title("GPT-2 Chatbot")
st.write("A simple chatbot built with GPT-2. Type your message below.")

#Input text box for user
user_input = st.text_input("You:")

if user_input:
    with st.spinner("Bit is typing..."):
        response = generate_response(user_input)
    st.write(f"Chatbot: {response}")
    

