from base import BaseAgent
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv("/Users/abhigyan/Downloads/project/Arctic/.env")
API_KEY = os.getenv("key")


# ----------------------
# Model Wrapper
# ----------------------
def openrouter_model(messages):
    """Wrapper for OpenRouter API with proper error handling."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen/qwen3-30b-a3b",
        "messages": messages,
        "temperature": 0.3,  # Lower temperature for more consistent structured output
        "response_format": {"type": "json_object"}  # Request JSON format if supported
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Try to parse as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If JSON parsing fails, return as string (BaseAgent will handle it)
            return content
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return {"error": f"API request failed: {str(e)}"}
    except (KeyError, IndexError) as e:
        print(f"Unexpected API response format: {e}")
        return {"error": "Unexpected API response format"}


# ----------------------
# Example Tools (Updated with proper signatures)
# ----------------------
def find_flight(source: str = None, destination: str = None):
    """
    Finds flight information from source to destination city.
    
    Args:
        source: Source city name
        destination: Destination city name
    """
    if source == "__describe__":
        return (
            "Finds flight information from source to destination city",
            '{"tool": "find_flight", "args": {"source": "New York", "destination": "Paris"}}'
        )
    
    if not source or not destination:
        return "Error: Both source and destination are required"
    
    return f"✈️ Flight found: {source} → {destination} | Airline: Air France | Departure: 10:00 AM | Arrival: 11:30 PM | Price: $850"


def hotel_booking(city: str = None, check_in: str = None, check_out: str = None):
    """
    Books a hotel in a city for given dates.
    
    Args:
        city: City name where hotel is needed
        check_in: Check-in date (YYYY-MM-DD format)
        check_out: Check-out date (YYYY-MM-DD format)
    """
    if city == "__describe__":
        return (
            "Books a hotel in a city for specified dates",
            '{"tool": "hotel_booking", "args": {"city": "Paris", "check_in": "2025-12-01", "check_out": "2025-12-05"}}'
        )
    
    if not all([city, check_in, check_out]):
        return "Error: city, check_in, and check_out are all required"
    
    # Calculate nights
    from datetime import datetime
    try:
        nights = (datetime.strptime(check_out, "%Y-%m-%d") - 
                 datetime.strptime(check_in, "%Y-%m-%d")).days
    except ValueError:
        nights = "?"
    
    return f"🏨 Hotel booked: Grand Hotel {city} | Check-in: {check_in} | Check-out: {check_out} | {nights} nights | Price: ${nights * 200 if isinstance(nights, int) else 'N/A'}"


def local_transport(city: str = None, mode: str = None):
    """
    Arranges local transportation in the city.
    
    Args:
        city: City name
        mode: Transport mode (taxi, bus, train, metro)
    """
    if city == "__describe__":
        return (
            "Arranges local transportation in a city",
            '{"tool": "local_transport", "args": {"city": "Paris", "mode": "metro"}}'
        )
    
    if not city or not mode:
        return "Error: Both city and mode are required"
    
    valid_modes = ["taxi", "bus", "train", "metro", "subway"]
    if mode.lower() not in valid_modes:
        return f"Error: Invalid mode '{mode}'. Choose from: {', '.join(valid_modes)}"
    
    prices = {"taxi": 50, "bus": 10, "train": 15, "metro": 12, "subway": 12}
    price = prices.get(mode.lower(), 15)
    
    return f"🚕 Transport arranged: {mode.capitalize()} pass in {city} | Valid for 3 days | Price: ${price}"


def weather_check(city: str = None, date: str = None):
    """
    Checks weather forecast for a city on a specific date.
    
    Args:
        city: City name
        date: Date to check weather (YYYY-MM-DD format)
    """
    if city == "__describe__":
        return (
            "Checks weather forecast for a city",
            '{"tool": "weather_check", "args": {"city": "Paris", "date": "2025-12-01"}}'
        )
    
    if not city:
        return "Error: city is required"
    
    # Mock weather data
    weather_options = [
        "☀️ Sunny, 22°C, perfect for sightseeing",
        "🌤️ Partly cloudy, 18°C, comfortable weather",
        "🌧️ Light rain expected, 15°C, bring an umbrella",
        "⛅ Cloudy, 20°C, good day for indoor activities"
    ]
    
    import hashlib
    weather_index = int(hashlib.md5(f"{city}{date}".encode()).hexdigest(), 16) % len(weather_options)
    return f"Weather in {city} on {date}: {weather_options[weather_index]}"


# ----------------------
# Initialize Agent
# ----------------------
agent = BaseAgent(
    model=openrouter_model,
    name="Agent",
    identity="Expert Travel Planning Agent specialized in creating comprehensive itineraries",
    instruction=(
        "You are a helpful travel agent. Your job is to:\n"
        "1. Use available tools to gather all necessary travel information\n"
        "2. Call tools in a logical sequence (flights first, then accommodation, then local transport)\n"
        "3. After gathering all information, present it as a well-organized itinerary\n"
        "4. Be thorough - use all relevant tools for the user's request\n"
        "5. Always verify you have all required information before finishing"
    ),
    task="Create comprehensive travel itineraries by coordinating flights, hotels, local transport, and weather information",
    tools={
        "find_flight": find_flight,
        "hotel_booking": hotel_booking,
        "local_transport": local_transport,
        "weather_check": weather_check
    },
    max_iterations=15,  # Increased for multi-step planning
    debug=True
)


# ----------------------
# Example Usage
# ----------------------
if __name__ == "__main__":
    print("=" * 80)
    print("TRAVEL PLANNING AGENT - DEMO")
    print("=" * 80)
    
    # Example 1: Full trip planning
    print("\n📋 TASK 1: Complete 3-day trip planning\n")
    result1 = agent.run(
        "Plan a 3-day trip from New York to Paris starting on 2025-12-01. "
        "I need flights, hotel for 3 nights, local metro pass, and weather info."
    )
    print("\n" + "=" * 80)
    print("FINAL ITINERARY:")
    print("=" * 80)
    print(result1)
    
    print("\n\n" + "=" * 80)
    
    # Example 2: Simpler request
    print("\n📋 TASK 2: Quick flight + hotel booking\n")
    result2 = agent.run(
        "I need a flight from London to Tokyo and a hotel from Dec 10 to Dec 15, 2025."
    )
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(result2)