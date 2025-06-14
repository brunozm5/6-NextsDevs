# Importa a biblioteca PyTorch para manipulação de tensores e computações do modelo
import torch
# Importa classes da biblioteca Transformers para carregar modelo e tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define a classe do chatbot psicológico
class PsicologoChatbot:
    # Método inicializador da classe
    def __init__(self):
        print("...")
        # Define o nome do modelo pré-treinado em português
        self.model_name = "pierreguillou/gpt2-small-portuguese"
        # Carrega o tokenizer correspondente ao modelo
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Carrega o modelo de linguagem causal pré-treinado
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # Define o contexto inicial (prompt) com instruções para o modelo
        self.contexto = (
            "Você é um psicólogo virtual que conversa com o usuário de forma acolhedora e reflexiva.\n"
            "Sempre responda como 'Psicólogo:' e nunca use nomes, cargos ou termos fora de contexto.\n"
            "Fale somente em português do Brasil.\n\n"
            "Exemplo de conversa:\n"
            "Usuário: Estou me sentindo mal ultimamente.\n"
            "Psicólogo: Sinto muito por isso. Você quer conversar mais sobre o que está te afetando?\n\n"
            "Histórico da conversa:\n"
        )
        # Inicializa uma lista vazia para armazenar o histórico da conversa
        self.conversa = []

    # Método para gerar a resposta do chatbot com base na mensagem do usuário
    def gerar_resposta(self, mensagem):
        # Adiciona a mensagem do usuário ao histórico, formatada com "Usuário:"
        self.conversa.append(f"Usuário: {mensagem}")
        # Constrói o prompt com o contexto, últimas 5 mensagens do histórico e prefixo "Psicólogo:"
        prompt = self.contexto + "\n".join(self.conversa[-5:]) + "\nPsicólogo:"
        # Tokeniza o prompt, convertendo em tensores com limite de 512 
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        # Desativa cálculo de gradientes para otimizar inferência
        with torch.no_grad():
            # Gera a resposta do modelo com parâmetros de controle
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=inputs["input_ids"].shape[1] + 60,  # Limita saída a 60  além da entrada
                temperature=0.8,  # Controla aleatoriedade
                top_p=0.95,  # Usa amostragem Top-p
                do_sample=True,  # Ativa amostragem estocástica
                pad_token_id=self.tokenizer.eos_token_id,  # Token de preenchimento
                eos_token_id=self.tokenizer.eos_token_id,  # Token de fim de sequência
            )
        # Decodifica a saída do modelo em texto, removendo tokens especiais
        resposta = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extrai apenas a fala do psicólogo após o último "Psicólogo:"
        if "Psicólogo:" in resposta:
            resposta = resposta.split("Psicólogo:")[-1]
        # Remove qualquer trecho após "Usuário:", se presente
        if "Usuário:" in resposta:
            resposta = resposta.split("Usuário:")[0]
        # Remove espaços em branco extras
        resposta = resposta.strip()
        # Substitui respostas muito curtas por uma padrão
        if len(resposta) < 10:
            resposta = "Você pode me contar um pouco mais sobre isso?"
        # Adiciona a resposta ao histórico, formatada com "Psicólogo:"
        self.conversa.append(f"Psicólogo: {resposta}")
        # Retorna a resposta gerada
        return resposta

# Verifica se o script é executado diretamente
if __name__ == "__main__":
    # Cria uma instância do chatbot
    chatbot = PsicologoChatbot()
    # Inicia loop interativo
    while True:
        # Solicita entrada do usuário
        mensagem = input("Você: ")
        # Verifica se o usuário quer sair
        if mensagem.lower() in ["sair", "exit", "quit"]:
            break
        # Gera e exibe a resposta do chatbot
        resposta = chatbot.gerar_resposta(mensagem)
        print(f"Psicólogo: {resposta}")
