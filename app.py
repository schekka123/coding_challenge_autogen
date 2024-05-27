import os
from dotenv import load_dotenv, find_dotenv

import chainlit as cl
import autogen
import pandas as pd
from typing_extensions import Annotated
from pandasql import sqldf

from agents.chainlit_agents import ChainlitAssistantAgent, ChainlitUserProxyAgent

load_dotenv(find_dotenv())

# -------------------- GLOBAL VARIABLES AND AGENTS ----------------------------------- # 
TRAVEL_AGENT_NAME = "Travel Agent"
SEARCH_AGENT_NAME = "Search Agent"
USER_PROXY_NAME = "User Proxy Agent"
EMPLOYEE_ID = ""
EVENT_NAME = ""

# -------------------- Config List. Edit to change your preferred model to use ----------------------------- # 
config_list = autogen.config_list_from_dotenv(
    dotenv_file_path='.env',
    model_api_key_map={
        "gpt-4o": "OPENAI_API_KEY",
    },
    filter_dict={
        "model": {
            "gpt-4o",
        }
    }
)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm_config = {"config_list": config_list, "api_key": OPENAI_API_KEY, "cache_seed": 42}

# -------------------- Load the Data -------------------------------------------- #
employees_df = pd.read_csv('data/employees.csv')
events_df = pd.read_csv('data/events.csv')
flights_df = pd.read_csv('data/flights.csv')

event_location_to_airport = {
    'Orlando': 'MCO',
    'Las Vegas': 'LAS',
    'Berlin': 'BER'
}

# -------------------- Define Functions -------------------------------------------- #

# Function to set the employee ID in the user session
def set_employee_id(employee_id: int):
    cl.user_session.set(EMPLOYEE_ID, employee_id)  # Store the employee ID in the session
    return f"Employee ID set to {employee_id}."

# Function to get the employee ID from the user session
def get_employee_id():
    employee_id = cl.user_session.get(EMPLOYEE_ID)  # Retrieve the employee ID from the session
    if not employee_id:
        return "Employee ID not found in session. Please set the Employee ID."
    return employee_id

# Function to set the event name in the user session
def set_event_name(event_name: str):
    cl.user_session.set(EVENT_NAME, event_name)  # Store the event name in the session
    return f"Event Name set to {event_name}."

# Function to get the event name from the user session
def get_event_name():
    event_name = cl.user_session.get(EVENT_NAME)  # Retrieve the event name from the session
    if not event_name:
        return "Event Name not found in session. Please set the Event Name."
    return event_name

# Function to prepare flight options based on the event name and days prior to the event
def prepare_flight_options(event_name: str, days_prior: int) -> list:
    employee_id = get_employee_id()  # Get the employee ID from the session
    if isinstance(employee_id, str):
        return [{"error": employee_id}]  # Return an error if employee ID is not found
    
    # Retrieve employee details
    employee = employees_df[employees_df['employee_id'] == employee_id]
    if employee.empty:
        return [{"error": f"Employee with ID '{employee_id}' not found."}]
    employee = employee.iloc[0]

    # Retrieve event details
    event = events_df[events_df['event_name'] == event_name]
    if event.empty:
        return [{"error": f"Event '{event_name}' not found."}]
    event = event.iloc[0]

    # Check if the event is in the same city as the employee's home office
    if employee['home_office'] == event['location']:
        return [{"message": f"No flight needed as the event '{event_name}' is in the same city as your home office ({employee['home_office']})."}]

    # Get the destination airport
    destination_airport = event_location_to_airport.get(event['location'])
    if not destination_airport:
        return [{"error": f"No airport mapping found for event location '{event['location']}'."}]

    all_flights = []

    # Search for flights over a range of days prior to the event
    for day in range(days_prior + 1):
        travel_date = pd.to_datetime(event['start_date']) - pd.Timedelta(days=day)

        # SQL query for direct flights
        query_direct = f"""
        SELECT *, 0 AS layovers FROM flights_df
        WHERE departure_airport = '{employee['preferred_airport']}'
        AND arrival_airport = '{destination_airport}'
        AND DATE(departure_time) = '{travel_date.strftime('%Y-%m-%d')}'
        """

        # SQL query for connecting flights
        query_connecting = f"""
        SELECT leg1.flight_id AS leg1_id, leg2.flight_id AS leg2_id,
               leg1.departure_airport AS departure_airport, leg2.arrival_airport AS arrival_airport,
               leg1.departure_time AS departure_time, leg2.arrival_time AS arrival_time,
               (leg1.cost + leg2.cost) AS total_cost,
               leg1.airline || ' -> ' || leg2.airline AS airline,
               1 AS layovers
        FROM flights_df leg1
        JOIN flights_df leg2 ON leg1.arrival_airport = leg2.departure_airport
        WHERE leg1.departure_airport = '{employee['preferred_airport']}'
        AND leg2.arrival_airport = '{destination_airport}'
        AND DATE(leg1.departure_time) = '{travel_date.strftime('%Y-%m-%d')}'
        AND DATETIME(leg2.departure_time) > DATETIME(leg1.arrival_time)
        """

        direct_flights = sqldf(query_direct, globals())
        connecting_flights = sqldf(query_connecting, globals())

        # Combine direct and connecting flights
        all_flights += direct_flights.to_dict(orient='records') + connecting_flights.to_dict(orient='records')

    if not all_flights:
        return [{"error": f"No available flights found for {employee['name']} to {event_name}."}]

    # Format the flight options
    for flight in all_flights:
        flight['departure_time'] = pd.to_datetime(flight['departure_time']).strftime('%B %d %Y, %I:%M %p %Z')
        flight['arrival_time'] = pd.to_datetime(flight['arrival_time']).strftime('%B %d %Y, %I:%M %p %Z')

    return all_flights

# Function to suggest the best flight based on user preference
def suggest_best_flight(flight_options: list, preference: str) -> dict:
    flight_options_df = pd.DataFrame(flight_options)  # Convert flight options to a DataFrame for easier manipulation
    if preference == 'price':
        best_flight = flight_options_df.loc[flight_options_df['cost'].idxmin()]  # Select the flight with the lowest cost
    elif preference == 'layovers':
        best_flight = flight_options_df.loc[flight_options_df['layovers'].idxmin()]  # Select the flight with the fewest layovers
    elif preference == 'airlines':
        best_flight = flight_options_df.loc[flight_options_df['airline'].idxmax()]  # Select the flight with the preferred airline (simplified logic)
    elif preference == 'travel time':
        best_flight = flight_options_df.loc[flight_options_df['departure_time'].idxmax()]  # Select the flight with the latest departure time (simplified logic)
    else:
        best_flight = flight_options_df.iloc[0]  # Default to the first option if preference is unclear
    return best_flight.to_dict()  # Return the best flight as a dictionary

# Function to list all events
def list_events() -> str:
    return events_df.to_string(index=False)  # Convert the events DataFrame to a string without the index

# Function to get details of a specific event
def event_details(event_name: str) -> str:
    event = events_df[events_df['event_name'] == event_name]  # Retrieve the event details from the DataFrame
    if event.empty:
        return f"Event '{event_name}' not found."  # Return an error message if the event is not found
    return event.iloc[0].to_string()  # Convert the event details to a string

# Function to get information of a specific employee
def get_employee_info(employee_id: int) -> str:
    employee = employees_df[employees_df['employee_id'] == employee_id]  # Retrieve the employee details from the DataFrame
    if employee.empty:
        return f"Employee with ID '{employee_id}' not found."  # Return an error message if the employee is not found
    return employee.iloc[0].to_string()  # Convert the employee details to a string

# Function to confirm and book a flight for an employee
def confirm_and_book_flight(flight_id: int) -> str:
    employee_id = get_employee_id()  # Get the employee ID from the session
    if isinstance(employee_id, str):
        return employee_id  # Return an error if the employee ID is not found
    
    print("Emp ID in confirm:", employee_id)  # Debugging print statement to check employee ID


    # Retrieve employee details
    employee = employees_df[employees_df['employee_id'] == employee_id]
    if employee.empty:
        return f"Employee with ID '{employee_id}' not found."
    employee = employee.iloc[0]

    # Retrieve flight details
    flight = flights_df[flights_df['flight_id'] == flight_id]
    if flight.empty:
        return f"Flight with ID '{flight_id}' not found."
    flight = flight.iloc[0]

    print("Flight details", flight)

    # Book the flight (simplified)
    new_booking = {
        'employee_id': employee_id,
        'employee_name': employee['name'],
        'flight_id': flight_id,
        'departure_airport': flight['departure_airport'],
        'arrival_airport': flight['arrival_airport'],
        'departure_time': flight['departure_time'],
        'arrival_time': flight['arrival_time'],
        'cost': flight['cost'],
        'layovers': flight.get('layovers', 0),  # Set default value of layovers to 0 if not present
        'airline': flight['airline']
    }

    bookings_df = pd.read_csv('data/bookings.csv')
    bookings_df = pd.concat([bookings_df, pd.DataFrame([new_booking])], ignore_index=True)
    print("Bookings_df", bookings_df)
    bookings_df.to_csv('data/bookings.csv', index=False)
    return f"Flight booked for {employee['name']} from {flight['departure_airport']} to {flight['arrival_airport']} on {flight['departure_time']}."

def find_round_trip_flights(event_name: str, days_prior: int, return_days_after: int) -> list:
    employee_id = get_employee_id()
    if isinstance(employee_id, str):
        return [{"error": employee_id}]
    
    # Retrieve employee details
    employee = employees_df[employees_df['employee_id'] == employee_id]
    if employee.empty:
        return [{"error": f"Employee with ID '{employee_id}' not found."}]
    employee = employee.iloc[0]

    # Retrieve event details
    event = events_df[events_df['event_name'] == event_name]
    if event.empty:
        return [{"error": f"Event '{event_name}' not found."}]
    event = event.iloc[0]

    # Check if the event is in the same city as the employee's home office
    if employee['home_office'] == event['location']:
        return [{"message": f"No flight needed as the event '{event_name}' is in the same city as your home office ({employee['home_office']})."}]

    # Get the destination airport
    destination_airport = event_location_to_airport.get(event['location'])
    if not destination_airport:
        return [{"error": f"No airport mapping found for event location '{event['location']}'."}]

    all_flights = []

    # Search for outgoing flights
    for day in range(days_prior + 1):
        travel_date = pd.to_datetime(event['start_date']) - pd.Timedelta(days=day)

        # SQL query for direct flights
        query_outgoing = f"""
        SELECT *, 0 AS layovers FROM flights_df
        WHERE departure_airport = '{employee['preferred_airport']}'
        AND arrival_airport = '{destination_airport}'
        AND DATE(departure_time) = '{travel_date.strftime('%Y-%m-%d')}'
        """
        outgoing_flights = sqldf(query_outgoing, globals()).to_dict(orient='records')

        for flight in outgoing_flights:
            flight['departure_time'] = pd.to_datetime(flight['departure_time']).strftime('%B %d %Y, %I:%M %p %Z')
            flight['arrival_time'] = pd.to_datetime(flight['arrival_time']).strftime('%B %d %Y, %I:%M %p %Z')

        all_flights.extend(outgoing_flights)

    round_trip_flights = []

    # Find matching return flights
    for flight in all_flights:
        for day in range(return_days_after + 1):
            return_date = pd.to_datetime(event['end_date']) + pd.Timedelta(days=day)

            query_return = f"""
            SELECT *, 0 AS layovers FROM flights_df
            WHERE departure_airport = '{destination_airport}'
            AND arrival_airport = '{employee['preferred_airport']}'
            AND DATE(departure_time) = '{return_date.strftime('%Y-%m-%d')}'
            """
            return_flights = sqldf(query_return, globals()).to_dict(orient='records')

            for return_flight in return_flights:
                return_flight['departure_time'] = pd.to_datetime(return_flight['departure_time']).strftime('%B %d %Y, %I:%M %p %Z')
                return_flight['arrival_time'] = pd.to_datetime(return_flight['arrival_time']).strftime('%B %d %Y, %I:%M %p %Z')
                round_trip_flights.append({'outgoing': flight, 'return': return_flight})

    if not round_trip_flights:
        return [{"error": f"No available round-trip flights found for {employee['name']} to {event_name}."}]
    
    return round_trip_flights


# -------------------- Instantiate agents at the start of a new chat. Call functions and tools the agents will use. ---------------------------- #
@cl.on_chat_start
async def on_chat_start():
    try:
        travel_agent = ChainlitAssistantAgent(
            name="Travel_Agent", llm_config=llm_config,
            system_message=(
                "You are a Travel Agent responsible for helping SAP employees book flights to SAP events."
                "Use the provided functions to handle travel-related queries:\n\n"
                "Functions Available:\n"
                "1. prepare_flight_options_handler(event_name, days_prior): Gather details and provide flight options based on user preferences.\n"
                "2. suggest_best_flight_handler(flight_options, preference): Suggest the best flight based on user preference (price, layovers, airlines, travel time).\n"
                "3. confirm_and_book_flight_handler(flight_id): Confirm the selected flight and book it.\n"
                "4. set_employee_id_handler(employee_id): Set the employee ID for the session.\n\n"
                "5. find_round_trip_flights_handler(event_name, days_prior, return_days_after): Find round-trip flights based on user preferences. ALWAYS RESPOND WITH THE FLIGHT ID WHEN DISPLAYING THE USER PREFERENCES AFTER CALLING THIS FUNCTION.\n\n"

                "If asked about anything regarding SAP, redirect the question to the Search Agent. Do not respond with TERMINATE when asked about SAP."
                "If asked about anything outside of your scope, redirect the question to the Search Agent."
                "If the Search_Agent or the User_Proxy_Agent has already asked the user for their details, do not ask for it again and regurgitate their response, instead say that the details are already set."
                "DO NOT REPLY WITH TERMINATE IF YOU ARE NOT THE EMPLOYEE ASSIGNED TO THE TASK."


                "If the employee ID is provided, do not ask for it again. Use the functions to navigate and complete the task efficiently. If the task is complete tell the User Proxy Agent that the task is completed."
            ),
            description="Travel Agent"
        )
        
        search_agent = ChainlitAssistantAgent(
            name="Search_Agent", llm_config=llm_config,
            system_message=(
                "You are a Search Agent responsible for retrieving SAP information. If asked about details outside of your scope, respond as a Helpful Assistant instead. Do not turn the user away. Your goal is to respond to the user with cordiality if so. Do not turn the user away and keep responding with the role of a helpful Assistant. "
                "Use the provided functions to handle information retrieval queries:\n\n"
                "Functions Available:\n"
                "1. list_events_handler(): List all upcoming SAP events.\n"
                "2. event_details_handler(event_name: str): Get details of a specific SAP event.\n"
                "3. get_employee_info_handler(employee_id: int): Get information of an employee based on employee ID.\n\n"

                "If asked about SAP details, respond and provide the details. If the task is complete, respond with 'TERMINATE'."

                "If the Travel_Agent or the User_Proxy_Agent has already asked the user for their details, do not ask for it again. Respond with 'TERMINATE'."
                "Respond to information retrieval queries using the provided functions. If the task is complete, respond with 'TERMINATE'."
            ),
            description="Search Agent"
        )
        
        user_proxy = ChainlitUserProxyAgent(
            name="User_Proxy_Agent",
            human_input_mode="TERMINATE",
            llm_config=llm_config,
            max_consecutive_auto_reply=2,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
            system_message="""
                Manage and execute tasks as directed by the Travel and Search Agents. Collaborate with both to achieve the task.
                Your role is to act as the user. If the Travel or Search Agent have already answered your question, do not regurgitate their response, instead respond with 'TERMINATE'.
                If the employeee info is asked, use the function: "get_employee_info_handler(employee_id: int): Get information of an employee based on employee ID."
                # Always use functions when describing the employee Info. Never make up any information that the user has not set. 

                # If the user asks to book a flight, if they have not yet provided the details, then ask them the following details:

                `What's your employee ID? Please provide your employee ID to proceed.`
                `What preferences do you have? (price, layovers, airlines, travel , round-trip)? Please provide your preferences to proceed.`
                `What's the event name? Please provide the event name to proceed.`

                If the Travel_Agent or the Search_Agent has already asked the user for their details, do not ask for it again and regurgitate. Respond with 'TERMINATE'.

                ALWAYS RESPOND WITH THE FLIGHT ID WHEN DISPLAYING THE USER PREFERENCES AFTER CALLING THIS FUNCTION.

                
                Respond to information retrieval queries using the provided functions. If the task is already complete, respond with 'TERMINATE'.
            """
        )
        
        cl.user_session.set(TRAVEL_AGENT_NAME, travel_agent)
        cl.user_session.set(SEARCH_AGENT_NAME, search_agent)
        cl.user_session.set(USER_PROXY_NAME, user_proxy)

        # Register the functions for LLM
        @user_proxy.register_for_execution()
        @travel_agent.register_for_llm(description="Gather details and provide flight options based on user preferences.")
        def prepare_flight_options_handler(event_name: Annotated[str, "Event Name"], days_prior: Annotated[int, "Days Prior"]) -> str:
            return prepare_flight_options(event_name, days_prior)

        @user_proxy.register_for_execution()
        @travel_agent.register_for_llm(description="Suggest the best flight based on user preference (price, layovers, airlines, travel time).")
        def suggest_best_flight_handler(flight_options: Annotated[list, "Flight Options"], preference: Annotated[str, "Preference"]) -> str:
            return suggest_best_flight(flight_options, preference)

        @user_proxy.register_for_execution()
        @search_agent.register_for_llm(description="List all upcoming SAP events.")
        def list_events_handler() -> str:
            return list_events()

        @user_proxy.register_for_execution()
        @search_agent.register_for_llm(description="Get details of a specific SAP event.")
        def event_details_handler(event_name: Annotated[str, "Event Name"]) -> str:
            return event_details(event_name)
        
        @user_proxy.register_for_execution()
        @search_agent.register_for_llm(description="Get information of an employee based on employee ID.")
        def get_employee_info_handler(employee_id: Annotated[int, "Employee ID"]) -> str:
            return get_employee_info(employee_id)
        
        @user_proxy.register_for_execution()
        @travel_agent.register_for_llm(description="Find round-trip flights based on user preferences.")
        def find_round_trip_flights_handler(event_name: Annotated[str, "Event Name"], days_prior: Annotated[int, "Days Prior"], return_days_after: Annotated[int, "Return Days After"]) -> str:
            return find_round_trip_flights(event_name, days_prior, return_days_after)


        # Register the function to set employee ID
        @user_proxy.register_for_execution()
        @travel_agent.register_for_llm(description="Set the employee ID for the session.")
        def set_employee_id_handler(employee_id: Annotated[int, "Employee ID"]) -> str:
            return set_employee_id(employee_id)

        # Register the functions for execution
        @user_proxy.register_for_execution()
        @travel_agent.register_for_llm(description="Confirm the selected flight and book it.")
        def confirm_and_book_flight_handler(flight_id: Annotated[int, "Flight ID"]) -> str:
            return confirm_and_book_flight(flight_id)

        msg = cl.Message(content=f"""Hello! What task would you like to get done today?""", 
                         author="Search Agent")
        await msg.send()

    except Exception as e:
        print("Error: ", e)
        pass


# -------------------- Run Conversation -------------------------------------------- #
@cl.on_message
async def run_conversation(message: cl.Message):
    print("Running conversation...")
    
    # Configuration for language model
    llm_config = {"config_list": config_list, "api_key": OPENAI_API_KEY, "cache_seed": 42}

    # Retrieve user message content
    user_message = message.content
    
    # Maximum number of iterations for the conversation
    MAX_ITER = 50
    
    # Retrieve agents from user session
    travel_agent = cl.user_session.get(TRAVEL_AGENT_NAME)
    search_agent = cl.user_session.get(SEARCH_AGENT_NAME)
    user_proxy = cl.user_session.get(USER_PROXY_NAME)

    # Initialize group chat with the agents
    groupchat = autogen.GroupChat(agents=[travel_agent, search_agent, user_proxy], messages=[], max_round=MAX_ITER,
                                  speaker_selection_method="round_robin")
    # Initialize the group chat manager
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Function to format messages based on agent type
    def format_message(agent_name, message_content):
        if agent_name == USER_PROXY_NAME:
            # For User Proxy Agent, format the message simply
            return f"message: '{message_content}'"
        else:
            # For other agents, include content and tool calls information
            content = message_content.get('content', '')
            tool_calls = message_content.get('tool_calls', [])
            tool_call_messages = ""
            for tool_call in tool_calls:
                function_name = tool_call['function']['name']
                tool_call_messages += f"\nTool {function_name} called."
            return f"{content}{tool_call_messages}"

    # Check the number of messages in the group chat
    if len(groupchat.messages) == 0:
        # If no messages yet, format and initiate the chat
        formatted_message = format_message(USER_PROXY_NAME, {'content': user_message})
        await cl.Message(content=f"""Starting agents on task...""").send()
        await cl.make_async(user_proxy.initiate_chat)(manager, message=formatted_message)
    elif len(groupchat.messages) < MAX_ITER:
        # If within the max iterations, format and send the message
        formatted_message = format_message(USER_PROXY_NAME, {'content': user_message})
        await cl.make_async(user_proxy.send)(manager, message=formatted_message)
    elif len(groupchat.messages) == MAX_ITER:
        # If max iterations reached, send an exit message
        await cl.make_async(user_proxy.send)(manager, message="exit")
