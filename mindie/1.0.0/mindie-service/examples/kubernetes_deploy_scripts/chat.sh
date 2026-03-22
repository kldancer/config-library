#!/bin/bash
curl -X POST --cacert xxxx  --cert xxxx --key xxxx -d '
{ 
	"model": "llama2_7b", 
	"messages": [{ 
 		"role": "user", 
  		"content": "You are a helpful assistant." 
 	}], 
 	"max_tokens": 200, 
 	"presence_penalty": 1.03, 
 	"frequency_penalty": 1.0, 
 	"seed": null, 
 	"temperature": 0.5, 
 	"top_p": 0.95, 
 	"stream": false 
}
' https://xx.xx.xx.xx:31015/v1/chat/completions 