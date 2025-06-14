import replicate

# Crie o cliente com seu token da API (troque 'SEU_TOKEN_AQUI' pelo seu token real)
client = replicate.Client(api_token="SEU_TOKEN_AQUI")

# Pegue o modelo e a versão que quer usar (exemplo: gpt-2 ou outro)
model = client.models.get("gpt2")  # ou outro modelo que você usar
version = model.versions.list()[0]  # pega a primeira versão disponível

def responder_mensagem(texto):
    # Chama o método predict com os parâmetros certos
    resposta = version.predict(
        inputs=texto,  # parâmetro correto depende do modelo, geralmente 'inputs' ou 'prompt'
        max_length=100,
        do_sample=True
    )
    return resposta
