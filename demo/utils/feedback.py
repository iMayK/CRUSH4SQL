import ipywidgets as widgets
from IPython.display import display

class Feedback:
    def __init__(self, question, predicted_sql, gc):
        self.question = question
        self.predicted_sql = predicted_sql
        self.gc = gc
        self.new_row = [question]
        self.options = [
            'Retrieval is wrong',
            'Retrieval is fine, but some other relevant table/columns are missing',
            'Retrieval is correct, but generated SQL is wrong',
            'Generated SQL seems correct, execution error',
            'Everything looks fine!'
        ]
        self.create_main_buttons()

    def create_buttons(self, options):
        return [self.create_button(option) for option in options]

    def create_button(self, description):
        button = widgets.Button(
            description=description,
            layout=widgets.Layout(width='35%', height='24px')
        )
        return button

    def create_main_buttons(self):
        self.main_buttons = self.create_buttons(self.options)
        for button in self.main_buttons:
            button.on_click(self.on_button_click)
        for button in self.main_buttons:
            display(button)

    def on_button_click(self, button):
        for b in self.main_buttons:
            b.layout.display = 'none'
        if button.description == self.options[0]:
            self.handle_incorrect_retrieval()
        elif button.description == self.options[1]:
            self.handle_incomplete_retrieval()
        elif button.description == self.options[2]:
            self.handle_incorrect_sql()
        elif button.description == self.options[3]:
            self.handle_incorrect_execution()
        else:
            self.handle_all_correct()

    def print_message(self, message):
        if hasattr(self, 'message_widget'):
            self.message_widget.close()
        self.message_widget = widgets.Label(value=message)
        display(self.message_widget)

    def submit_response(self):
        worksheet = self.gc.open_by_url('https://docs.google.com/spreadsheets/d/1PTiGJcXDntJNPVjkFRdgSersW4HqkICQtDc7zezr6w8/edit#gid=0').sheet1
        rows = worksheet.get_all_values()
        last_row = len(rows)
        for i, feedback in enumerate(self.new_row):
            worksheet.update_cell(last_row + 1, i + 1, feedback)

    def handle_incorrect_retrieval(self):
      self.print_message('Can you provide correct table(s)?')
      self.correct_table_yes_no_buttons = self.create_buttons(['yes', 'no'])
      for button in self.correct_table_yes_no_buttons:
          display(button)

      def on_correct_table_submit_button_clicked(b):
          correct_tables = self.correct_table_input_area.value
          self.correct_table_input_area.layout.display = 'none'
          self.correct_table_submit_button.layout.display = 'none'
          self.new_row.extend(['yes', correct_tables, 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
          self.print_message('Thanks for the feedback!')
          self.submit_response()

      def on_correct_table_yes_no_button_clicked(b):
          status = b.description
          for button in self.correct_table_yes_no_buttons:
              button.layout.display = 'none'
          if status == 'yes':
              self.correct_table_input_area = widgets.Textarea(
                  value='',
                  placeholder='Type relevant table names/codes',
                  description='',
                  rows=3,
                  layout=widgets.Layout(width="auto", height="auto")
              )
              self.correct_table_submit_button = widgets.Button(description='Submit')
              self.correct_table_submit_button.on_click(on_correct_table_submit_button_clicked)
              display(self.correct_table_input_area, self.correct_table_submit_button)
          else:
              self.new_row.extend(['yes', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'])
              self.print_message('Thanks for the feedback!')
              self.submit_response()

      for button in self.correct_table_yes_no_buttons:
          button.on_click(on_correct_table_yes_no_button_clicked)

    def handle_incomplete_retrieval(self):
        self.print_message('Can you provide additional relevant table(s)?')
        self.extra_table_yes_no_buttons = self.create_buttons(['yes', 'no'])
        for button in self.extra_table_yes_no_buttons:
            display(button)

        def on_extra_table_submit_button_clicked(b):
            extra_tables = self.extra_table_input_area.value
            self.extra_table_input_area.layout.display = 'none'
            self.extra_table_submit_button.layout.display = 'none'
            self.new_row.extend(['no', 'NA', 'yes', extra_tables])
            self.ask_about_sql_correctness()

        def on_extra_table_yes_no_button_clicked(b):
            status = b.description
            for button in self.extra_table_yes_no_buttons:
                button.layout.display = 'none'
            if status == 'yes':
                self.extra_table_input_area = widgets.Textarea(
                    value='',
                    placeholder='Type additional relevant tables',
                    description='',
                    rows=3,
                    layout=widgets.Layout(width="auto", height="auto")
                )
                self.extra_table_submit_button = widgets.Button(description='Submit')
                self.extra_table_submit_button.on_click(on_extra_table_submit_button_clicked)
                display(self.extra_table_input_area, self.extra_table_submit_button)
            else:
                self.new_row.extend(['no', 'NA', 'yes', 'NA'])
                self.ask_about_sql_correctness()

        for button in self.extra_table_yes_no_buttons:
            button.on_click(on_extra_table_yes_no_button_clicked)

    def ask_about_sql_correctness(self):
        self.print_message('Is the SQL correct based on the retrieval?')
        self.sqlcorrect_yes_no_buttons = self.create_buttons(['yes', 'no', 'not sure'])
        for button in self.sqlcorrect_yes_no_buttons:
            display(button)

        def on_sqlcorrect_yes_no_button_clicked(b):
            status = b.description
            for button in self.sqlcorrect_yes_no_buttons:
                button.layout.display = 'none'
            if status == 'yes':
                self.new_row.extend(['no', self.predicted_sql, 'NA', 'NA'])
                self.print_message('Thanks for the feedback!')
                self.submit_response()
            else:
                self.handle_incorrect_sql()

        for button in self.sqlcorrect_yes_no_buttons:
            button.on_click(on_sqlcorrect_yes_no_button_clicked)

    def handle_incorrect_sql(self):
        self.print_message('Want to register correct SQL (you can edit the predicted one as well)?')
        self.sql_input_yes_no_buttons = self.create_buttons(['yes', 'no'])
        for button in self.sql_input_yes_no_buttons:
            display(button)

        def on_sql_submit_button_clicked(b):
            correct_sql = self.sql_input_area.value
            self.sql_input_area.layout.display = 'none'
            self.sql_submit_button.layout.display = 'none'
            if len(self.new_row) == 1:
                self.new_row.extend(['no', 'NA', 'no', 'NA'])
            self.new_row.extend(['yes', correct_sql, 'NA', 'NA'])
            self.print_message('Thanks for the feedback!')
            self.submit_response()

        def on_sql_input_yes_no_button_clicked(b):
            status = b.description
            for button in self.sql_input_yes_no_buttons:
                button.layout.display = 'none'

            if status == 'no':
                if len(self.new_row) == 1:
                    self.new_row.extend(['no', 'NA', 'no', 'NA'])
                self.new_row.extend(['yes', 'NA', 'NA', 'NA'])
                self.print_message('Thanks for the feedback!')
                self.submit_response()
            else:
                self.sql_input_area = widgets.Textarea(
                    value=self.predicted_sql,
                    placeholder='Type correct SQL',
                    description='',
                    rows=5,
                    layout=widgets.Layout(width="auto", height="auto")
                )
                self.sql_submit_button = widgets.Button(description='Submit')
                self.sql_submit_button.on_click(on_sql_submit_button_clicked)
                display(self.sql_input_area, self.sql_submit_button)

        for button in self.sql_input_yes_no_buttons:
            button.on_click(on_sql_input_yes_no_button_clicked)

    def handle_incorrect_execution(self):
        self.new_row.extend(['no', 'NA', 'no', 'NA', 'no', self.predicted_sql, 'yes', 'NA'])
        self.print_message('Thanks for the feedback!')
        self.submit_response()

    def handle_all_correct(self):
        self.new_row.extend(['no', 'NA', 'no', 'NA', 'no', self.predicted_sql, 'no', 'all_good'])
        self.print_message('Thanks for the feedback!')
        self.submit_response()
