# frozen_string_literal: true

Rails.application.routes.draw do
  root 'emps#index'

  resource :emp, only: [:create]
end
