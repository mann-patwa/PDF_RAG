from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .llm_engine import ask_llm


def home(request):
    return render(request, "index.html")


@csrf_exempt
def ask(request):
    if request.method == "POST":
        query = request.POST.get("query", "")
        answer = ask_llm(query)
        return JsonResponse({"reply": answer})
