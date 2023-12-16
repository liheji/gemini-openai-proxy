package protocol

import (
	"fmt"
	"time"

	"github.com/google/generative-ai-go/genai"
	openai "github.com/sashabaranov/go-openai"

	"github.com/zhu327/gemini-openai-proxy/pkg/util"
)

type CompletionChoice struct {
	Index int `json:"index"`
	Delta struct {
		Content string `json:"content"`
	} `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
}

func GenaiResponseToStreamComplitionResponse(
	genaiResp *genai.GenerateContentResponse,
	respID string,
	created int64,
) *CompletionResponse {
	resp := CompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", respID),
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   "gemini-pro",
		Choices: make([]CompletionChoice, 0, len(genaiResp.Candidates)),
	}

	for i, candidate := range genaiResp.Candidates {
		var content string
		if len(candidate.Content.Parts) > 0 {
			if s, ok := candidate.Content.Parts[0].(genai.Text); ok {
				content = string(s)
			}
		}

		var reason = string(openai.FinishReasonNull)
		if candidate.FinishReason == 1 {
			reason = string(openai.FinishReasonStop)
		}

		choice := CompletionChoice{
			Index:        i,
			FinishReason: &reason,
		}
		choice.Delta.Content = content

		resp.Choices = append(resp.Choices, choice)
	}
	return &resp
}

func GenaiResponseToOpenaiResponse(
	genaiResp *genai.GenerateContentResponse,
) openai.ChatCompletionResponse {
	resp := openai.ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", util.GetUUID()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "gemini-pro",
		Choices: make([]openai.ChatCompletionChoice, 0, len(genaiResp.Candidates)),
	}

	for i, candidate := range genaiResp.Candidates {
		var content string
		if len(candidate.Content.Parts) > 0 {
			if s, ok := candidate.Content.Parts[0].(genai.Text); ok {
				content = string(s)
			}
		}

		choice := openai.ChatCompletionChoice{
			Index: i,
			Message: openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: content,
			},
			FinishReason: openai.FinishReasonStop,
		}
		resp.Choices = append(resp.Choices, choice)
	}
	return resp
}

func SetGenaiChatByOpenaiRequest(cs *genai.ChatSession, req openai.ChatCompletionRequest) {
	cs.History = make([]*genai.Content, 0, len(req.Messages))
	if len(req.Messages) > 1 {
		for _, message := range req.Messages[:len(req.Messages)-1] {
			switch message.Role {
			case openai.ChatMessageRoleSystem:
				cs.History = append(cs.History, []*genai.Content{
					{
						Parts: []genai.Part{
							genai.Text(message.Content),
						},
						Role: "user",
					},
					{
						Parts: []genai.Part{
							genai.Text("ok."),
						},
						Role: "model",
					},
				}...)
			case openai.ChatMessageRoleAssistant:
				cs.History = append(cs.History, &genai.Content{
					Parts: []genai.Part{
						genai.Text(message.Content),
					},
					Role: "model",
				})
			case openai.ChatMessageRoleUser:
				cs.History = append(cs.History, &genai.Content{
					Parts: []genai.Part{
						genai.Text(message.Content),
					},
					Role: "user",
				})
			}
		}
	}
}

func SetGenaiModelByOpenaiRequest(model *genai.GenerativeModel, req openai.ChatCompletionRequest) {
	if req.MaxTokens != 0 {
		model.MaxOutputTokens = int32(req.MaxTokens)
	}
	if req.Temperature != 0 {
		model.Temperature = req.Temperature
	}
	if req.TopP != 0 {
		model.TopP = req.TopP
	}
	model.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockOnlyHigh,
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockOnlyHigh,
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockOnlyHigh,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockOnlyHigh,
		},
	}
}