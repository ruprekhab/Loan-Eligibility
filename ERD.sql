-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- Link to schema: https://app.quickdatabasediagrams.com/#/d/3zgMBG
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE "loan" (
    "age" numeric   NOT NULL,
    "income" numeric   NOT NULL,
    "home_ownership" Varchar(10)   NOT NULL,
    "employment_duration" numeric   NOT NULL,
    "loan_purpose" Varchar(30)   NOT NULL,
    "loan_grade" Varchar(3)   NOT NULL,
    "loan_amount" numeric   NOT NULL,
    "int_rate" numeric   NOT NULL,
    "loan_status" numeric   NOT NULL,
    "loan_income_pct" float   NOT NULL,
    "past_default_status" varchar(2)   NOT NULL,
    "credit_history_length" numeric   NOT NULL
);

