# DevPulse AI

ავტონომიური დეველოპერის ასისტენტი: LangGraph მართავს მრავალეტაპიან სამუშაოს, GitHub მონაცემები მიიღება MCP სერვერიდან, ვებ-ძიება მუშაობს Tavily-ით ან გასაღების გარეშე Google News RSS-ით (Hacker News / Algolia სარეზერვო წყაროა), ხოლო ელფოსტა და Slack შეტყობინებები იგზავნება მხოლოდ მომხმარებლის აშკარა თანხმობის შემდეგ.

## შესაძლებლობები

- ღია GitHub Issue-ები, ბოლო კომიტები, დამერჯილი PR-ები და ცალკეული PR-ის დეტალები.
- ტექნოლოგიური სიახლეების Tavily ძიება (`TAVILY_API_KEY`-ით), გასაღების გარეშე Google News RSS fallback-ით და Hacker News / Algolia სარეზერვო წყაროთი.
- მომხმარებლის ელფოსტისა და `owner/repo` პრეფერენციების დამახსოვრება LangGraph Store-ში.
- ელფოსტის დეტალური პრევიუ და `yes/no` Human-in-the-Loop დამტკიცება.
- Slack Incoming Webhook-ის საშუალებით შეტყობინების გაგზავნა, იგივე `yes/no` Human-in-the-Loop დაცვით.
- `eval.py`-ში Tool Selection Accuracy და LLM-as-a-Judge Faithfulness შეფასება სატესტო მაგალითებით.

## გაშვება

1. შექმენით ვირტუალური გარემო და დააინსტალირეთ `requirements.txt`.
2. დააკოპირეთ `.env.example` როგორც `.env`, უფასო Gemini API key შექმენით [Google AI Studio-ში](https://aistudio.google.com/apikey) და შეავსეთ `GOOGLE_API_KEY`. Gemini გასაღები საჭიროა აგენტის LLM პასუხებისა და `eval.py`-ს გასაშვებად. GitHub token საჯარო API-ის rate limit-ს ზრდის; Tavily გასაღები არასავალდებულოა, რადგან გასაღების გარეშე Hacker News / Algolia გამოიყენება.
3. გაუშვით `python main.py`.

`agent.py` GitHub MCP server-ის ინსტრუმენტებს MCP Python Client-ით, stdio ტრანსპორტით იძახებს; server-ის ცალკე გაშვება საჭირო არ არის. სურვილის შემთხვევაში server დამოუკიდებლადაც გაუშვით: `python mcp_server.py`.

SMTP ელფოსტის რეალური გაგზავნისთვის დააყენეთ `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD` და `SMTP_FROM`. `465` პორტი იყენებს SSL-ს, ხოლო `587` — STARTTLS-ს. Gmail-ისთვის გამოიყენეთ Google App Password და არა ძირითადი პაროლი. გაგზავნამდე აგენტი აჩვენებს მიმღებს, თემასა და სრულ ტექსტს და ელოდება `yes/no` პასუხს. SMTP პაროლი რეპოზიტორიაში არ შეინახოთ.

Slack-ის გასააქტიურებლად შექმენით Slack App Incoming Webhook, მიუთითეთ `.env`-ში `SLACK_WEBHOOK_URL` და სურვილისამებრ `SLACK_CHANNEL_NAME` (პრევიუში არგუმენტად ნაჩვენები არხის სახელი). Webhook მიბმულია Slack-ში არჩეულ არხზე. URL საიდუმლოდ შეინახეთ. Slack შეტყობინების გაგზავნამდეც აგენტი აჩვენებს სრულ ტექსტს და ითხოვს თანხმობას.

## შეფასება

`python eval.py` აფასებს ინსტრუმენტების არჩევას და LLM-ით ამოწმებს, ემყარება თუ არა digest მოწოდებულ GitHub მტკიცებულებებს. შეფასება სატესტო მონაცემებზე მუშაობს და არ იძახებს რეალურ GitHub ან ელფოსტის ინსტრუმენტებს. საკუთარი შედეგისთვის გამოიყენეთ `faithfulness_judge(digest, github_evidence)`.

## მეხსიერება

LangGraph `InMemoryStore` გამოიყენება მიმდინარე პროცესში, ხოლო მომხმარებლის პრეფერენციები დამატებით `devpulse_memory.sqlite3` ფაილშიც ინახება და აპის შემდეგ გაშვებაზეც იტვირთება. მონაცემთა ბაზა git-ში იგნორირებულია.
