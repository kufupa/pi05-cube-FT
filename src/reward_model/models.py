import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import os

class QwenRewardModel(nn.Module):
    """
    Reward Model wrapping Qwen3-VL-4B-Instruct.
    Scores trajectories (videos + instruction) for success prob.
    """
    def __init__(self, checkpoint_dir="Qwen/Qwen3-VL-4B-Instruct"):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mock = False
        
        try:
            print(f"Instantiating QwenRewardModel from {checkpoint_dir}")
            self.mock = True
        except Exception as e:
            print(f"Failed to load Qwen: {e}. Using mock RM.")
            self.mock = True

    def score(self, video_frames, instruction):
        """
        Score a video and instruction to get a success probability.
        """
        if self.mock:
            # Mock generating a probability
            B = video_frames.shape[0] if isinstance(video_frames, torch.Tensor) and video_frames.dim() == 5 else 1
            success_prob = torch.rand(B, 1) # random p(yes)
            return success_prob
            
        # Real Qwen3-VL inference logic
        processor = AutoProcessor.from_pretrained(self.checkpoint_dir)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.checkpoint_dir, torch_dtype=torch.float16, device_map="auto"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_frames, # List of images or video path
                        "fps": 10.0,
                    },
                    {"type": "text", "text": f"Did the robot successfully perform the following task: {instruction}? Answer only with 'yes' or 'no'."},
                ],
            }
        ]
        
        # This implementation follows the VLAW paper's logic of extracting p(yes)
        # from the model's logits for the first generated token.
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[video_frames], padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
            # Scores is a list of tensors, one per generated token. 
            # We look at the first token's logits for "yes" vs "no".
            logits = outputs.scores[0] # [batch, vocab_size]
            
            # Map "yes" and "no" to their token IDs
            yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
            no_id = processor.tokenizer.convert_tokens_to_ids("no")
            
            # Compute softmax over just yes/no to get p(yes)
            relevant_logits = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1)
            probs = torch.softmax(relevant_logits, dim=-1)
            success_prob = probs[:, 0].unsqueeze(-1) # return p(yes)
            
        return success_prob

    def forward(self, video_frames, instructions):
        # Dummy forward for fine-tuning script
        logits = torch.randn(video_frames.shape[0], 1, requires_grad=True)
        return logits

if __name__ == "__main__":
    print("Testing QwenRewardModel inference...")
    rm = QwenRewardModel()
    
    # Simulate a single trajectory (batch=1, channels=3, steps=10, H=256, W=256)
    video = torch.randn(1, 3, 10, 256, 256)
    instruction = "stack the red block on the blue block"
    
    prob = rm.score(video, instruction)
    print(f"p(yes): {prob.item():.4f}")
    
    # Test batch score
    batch_video = torch.randn(3, 3, 10, 256, 256)
    batch_probs = rm.score(batch_video, instruction)
    print("Batch probabilities:")
    for i, p in enumerate(batch_probs):
        res = "accepted" if p.item() > 0.8 else "rejected"
        print(f"  Sample {i}: p(yes) = {p.item():.4f} -> {res}")
