import streamlit as st
import streamlit_authenticator as stauth
import yaml

def app():
    
    with open('config/auth.yaml') as file:
        config_auth = yaml.load(file, Loader=yaml.SafeLoader)

    authenticator = stauth.Authenticate(
        config_auth['credentials'],
        config_auth['cookie']['name'],
        config_auth['cookie']['key'],
        config_auth['cookie']['expiry_days'],
        config_auth['preauthorized']
    )

    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{name}*')

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
    
    # if st.button("Register"):
    try:
        if authenticator.register_user('Register user', preauthorization=False):
            st.success('User registered successfully')
    except Exception as e:
        st.error(e)

    with open('config/auth.yaml', 'w') as file:
        yaml.dump(config_auth, file, default_flow_style=False)