from agent.llm import base_endpoint

def test_endpoint_with_portal_path_is_reduced_to_host():
    # Azure portal shows full URLs; the SDK needs only scheme + host.
    assert base_endpoint("https://x.services.ai.azure.com/openai/v1/responses") == "https://x.services.ai.azure.com/"
    assert base_endpoint("https://x.openai.azure.com/") == "https://x.openai.azure.com/"
    assert base_endpoint(" https://x.openai.azure.com ") == "https://x.openai.azure.com/"
