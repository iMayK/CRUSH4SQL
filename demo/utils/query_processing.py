import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

from .utils import ndap_pipeline

from .feedback import Feedback

class QueryProcessingSession:  
    def __init__(self):  
        self.wants_to_submit_status = None  
        self.output_type = None  
        self.gc = None  
        self.api_type = 'python'
        self.api_key = None
        self.endpoint = None
        self.api_version = None
        self.correct_txt_sql_pairs = {}
        self.output_options = [
            'hallucinated schema | most relevant tables & columns | SQL',  
            'most relevant tables & columns | SQL',  
            'SQL' 
        ]

    def ask_openai_creds(self):
        api_message_widget = widgets.HTML(  
            value=f'<p><b>Please enter OpenAI key</b></p>'  
        )  
        display(api_message_widget)  

        api_key_widget = widgets.Password(  
            placeholder='Enter API Key',  
            description='API Key:',  
            disabled=False  
        )  
        cred_submit_button = widgets.Button(description='Submit')

        def on_cred_submit_button_clicked(b):
            self.api_key = api_key_widget.value  
            clear_output()
            if self.wants_to_submit_status is None:
                self.set_submit_status()

        cred_submit_button.on_click(on_cred_submit_button_clicked)
        display(api_key_widget, cred_submit_button)

    def set_submit_status(self):  
        wants_to_submit_widget = widgets.HTML(  
            value=f'<p><b>Would you like to submit feedback? </b> <font color="#FF0000">(NOTE: this will require google authentication)</font></p>'  
        )  
        display(wants_to_submit_widget)  
      
        wants_to_submit_yes_no_buttons = [  
            widgets.Button(  
                description=option,  
                    layout=widgets.Layout(width='35%', height='24px')  
                )  
            for option in ['yes', 'no']  
        ]  
        for button in wants_to_submit_yes_no_buttons:  
            display(button)  
      
        def on_wants_to_submit_yes_no_button_clicked(b):  
            self.wants_to_submit_status = b.description  
      
            if self.wants_to_submit_status == 'yes':  
                from google.colab import auth  
                auth.authenticate_user()  
                import gspread  
                from google.auth import default  
      
                creds, _ = default()  
                self.gc = gspread.authorize(creds)  

            clear_output()
            if self.output_type is None:
                self.set_output_type()
      
        for button in wants_to_submit_yes_no_buttons:  
            button.on_click(on_wants_to_submit_yes_no_button_clicked)  

    def set_output_type(self):  
        output_log_message_widget = widgets.HTML(  
            value=f'<p><b>Please select the output type</b></p>'  
        )  
        display(output_log_message_widget)  

        output_log_buttons = [  
            widgets.Button(  
                description=option,  
                    layout=widgets.Layout(width='35%', height='24px')  
                )  
            for option in self.output_options  
        ]  
        for button in output_log_buttons:  
            display(button)  

        def on_output_log_button_clicked(b):  
            self.output_type = b.description  

            clear_output()
            self.ask_question()
            # output_log_message_widget.layout.display = 'none'  
            # for button in output_log_buttons:  
            #     button.layout.display = 'none'  

        for button in output_log_buttons:  
            button.on_click(on_output_log_button_clicked)  

    def ask_question(self):  
        question_input_area = widgets.Textarea(
            value='',
            placeholder='Type your question',
            description='',
            rows=3,
            layout=widgets.Layout(width="auto")
        )
    
        question_submit_button = widgets.Button(description='Submit')

        def on_question_submit_button_clicked(b):
            question = question_input_area.value
            question_input_area.layout.display = 'none'
            question_submit_button.layout.display = 'none'

            question_widget = widgets.HTML(
                value=f'<p><b>QUESTION:</b></p><pre style="background-color: #F5F5F5; padding: 0.5em;">{question}</pre>'
            )
            display(question_widget)

            try:
                hallucinated_schema, predicted_schema, predicted_sql = ndap_pipeline(
                    question, self.api_type, self.api_key, self.endpoint, self.api_version, self.correct_txt_sql_pairs
                )

                if self.output_type == self.output_options[0]:
                    hallucinated_schema = "\n".join(hallucinated_schema)
                    hallucinated_schema_widget = widgets.HTML(
                        value=f'<p><b>HALLUCINATED_SCHEMA:</b></p><pre style="background-color: #F5F5F5; padding: 0.5em;">{hallucinated_schema}</pre>'
                    )
                    display(hallucinated_schema_widget)

                if self.output_type in self.output_options[:2]:
                    predicted_schema_widget = widgets.HTML(
                        value=f'<p><b>RELEVANT TABLES:</b></p><pre style="background-color: #F5F5F5; padding: 0.5em;">{predicted_schema}</pre>'
                    )
                    display(predicted_schema_widget)

                sql_widget = widgets.HTML(
                    value=f'<p><b>SQL:</b></p><pre style="background-color: #F5F5F5; padding: 0.5em;">{predicted_sql}</pre>'
                )
                display(sql_widget)

                if self.wants_to_submit_status == 'yes':
                    feedback_message_widget = widgets.HTML(
                        value=f'<p><b><br>Want to submit feedback for this question?</b></p>'
                    )
                    display(feedback_message_widget)

                    feedback_yes_no_buttons = [
                        widgets.Button(
                            description=option,
                                layout=widgets.Layout(width='35%', height='24px')
                            )
                        for option in ['yes', 'no']
                    ]
                    for button in feedback_yes_no_buttons:
                        display(button)

                    def on_feedback_yes_no_button_clicked(b):
                        status = b.description

                        feedback_message_widget.layout.display = 'none'
                        for button in feedback_yes_no_buttons:
                            button.layout.display = 'none'

                        if status == 'yes':
                            feedback_widget = widgets.HTML(
                              value=f'<p><b><br>Feedback (please click appropriate option):</b></p>'
                            )
                            display(feedback_widget)
                            feedback = Feedback(question, predicted_sql, self.gc)

                    for button in feedback_yes_no_buttons:
                        button.on_click(on_feedback_yes_no_button_clicked)
            except Exception as e:
                error_message_widget = widgets.HTML(
                    value=f'<p><b><br>An error occurred while processing the question</b></p>'
                )
                display(error_message_widget)

        question_submit_button.on_click(on_question_submit_button_clicked)
        display(question_input_area, question_submit_button)
