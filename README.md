# Chatty-bot
A chat-bot using an attention Encoder-Decoder with GloVe embeddings


RNN seq2seq model with Attention Tensor breakdown

***********************************************************************************************************************************************************************************************************************************************
**ENCODER**

Input --> B x T x N_in

Embedding --> 

	Load weights from GloVe (B x T x H_emb)


Output --> 

	Encoder_output (B x T x H)
	Hidden (B x 1 x H)


***************************************************************************************************************************************************************************************************************************************************
**ATTENTION**

Input --> 
	
	Encoder_output (B x T x H)
	Hidden (B x 1 x H)

Output --> 

	attn_weights (B x T)

***************************************************************************************************************************************************************************************************************************************************
**DECODER**

Input --> 
	
	Decoder_input (B x 1 x N_out)
	Last_hidden (B x 1 x H)

Embedding --> 

	Load weights from GloVe (B x T x H_emb)

Context -->

	attn_weights x Encoder_outputs (B x 1 x N_out)	



Output --> B x N_out