curl -H "Content-type: application/json" -X POST --cacert xxxx  --cert xxxx --key xxxx -d '
{
    "inputs": "My name is Olivier and I",
    "parameters": {
        "decoder_input_details": false,
        "details": true,
        "do_sample": true,
        "max_new_tokens": 20,
        "repetition_penalty": 1.03,
        "return_full_text": false,
        "seed": null,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.95,
        "truncate": null,
        "typical_p": 0.5,
        "watermark": false
    }
}' https://xxx.xxx.xxx.xxx:31015/generate_stream