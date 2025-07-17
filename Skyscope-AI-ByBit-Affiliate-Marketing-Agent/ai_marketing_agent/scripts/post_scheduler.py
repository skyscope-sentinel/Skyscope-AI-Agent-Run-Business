import os
import yaml # To load settings
from datetime import datetime
import getpass # For password input if not using rich.prompt fully

# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

# Initialize Rich Console
console = Console()

# Attempt to import Google API client libraries
try:
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google.oauth2.credentials import Credentials as UserCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    import google.auth.transport.requests
=======

# Attempt to import Google API client libraries
try:
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials # For service account auth
    from google.oauth2.credentials import Credentials as UserCredentials # For user OAuth
    from google_auth_oauthlib.flow import InstalledAppFlow # For user OAuth
    import google.auth.transport.requests # For refreshing credentials
 main
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_LIBS_AVAILABLE = True
except ImportError:
    console.print("[bold yellow]POST_SCHEDULER_WARNING:[/bold yellow] google-api-python-client or google-auth libraries not found. Blogger posting will be disabled. Please install them: [i]pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib[/i]")
    GOOGLE_LIBS_AVAILABLE = False

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/settings.yaml')
=======
    print("POST_SCHEDULER_WARNING: google-api-python-client or google-auth libraries not found. Blogger posting will be disabled. Please install them: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    GOOGLE_LIBS_AVAILABLE = False

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/settings.yaml') # Path relative to this script
 main

def load_settings():
    """Loads settings from the YAML configuration file."""
    if not os.path.exists(CONFIG_PATH):
        console.print(f"[bold red]POST_SCHEDULER_ERROR:[/bold red] Configuration file not found at {CONFIG_PATH}. Cannot proceed.")
=======
        print(f"POST_SCHEDULER_ERROR: Configuration file not found at {CONFIG_PATH}. Cannot proceed.")
 main
        return None
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            console.print(f"[bold red]POST_SCHEDULER_ERROR:[/bold red] Configuration file {CONFIG_PATH} is empty or malformed.")
            return None
        return config
    except Exception as e:
        console.print(f"[bold red]POST_SCHEDULER_ERROR:[/bold red] Error loading configuration from {CONFIG_PATH}: {e}")
        return None

def prompt_for_credentials(platform_name):
    """
    Prompts the user for one-time credentials for a given platform.
    Returns a dict with 'username' and 'password' or None.
    """
    console.print(Panel(f"Missing Credentials for {platform_name}", title="[yellow]Action Required[/yellow]", expand=False, border_style="yellow"))
    console.print(f"[yellow]Posting to '{platform_name}' requires credentials not found in your current configuration (or they failed).[/yellow]")
    console.print("[bold orange_red1]SECURITY WARNING: Entering credentials directly is for one-time use only and carries security risks. For regular use, configure credentials securely (e.g., via environment variables as specified in settings.yaml).[/bold orange_red1]")

    if not Confirm.ask(f"Do you want to provide credentials for '{platform_name}' for this session?", default=False, console=console):
        console.print(f"[blue]INFO:[/blue] Skipping credential entry for '{platform_name}'.")
        return None

    username = Prompt.ask(f"Enter username/email for '{platform_name}'", console=console)
    password = Prompt.ask(f"Enter password for '{platform_name}'", password=True, console=console)

    if username and password:
        console.print(f"[green]Credentials received for '{platform_name}' for this session.[/green]")
        return {"username": username, "password": password}
    else:
        console.print(f"[yellow]WARN:[/yellow] No username or password provided for '{platform_name}'.")
=======
            print(f"POST_SCHEDULER_ERROR: Configuration file {CONFIG_PATH} is empty or malformed.")
            return None
        return config
    except Exception as e:
        print(f"POST_SCHEDULER_ERROR: Error loading configuration from {CONFIG_PATH}: {e}")
 main
        return None

# --- Blogger Integration ---
def get_blogger_service(settings):
    if not GOOGLE_LIBS_AVAILABLE:
        console.print("[blue]POST_SCHEDULER_INFO:[/blue] Google client libraries not available. Cannot get Blogger service.")
=======
    """
    Authenticates and returns the Blogger API service client.
    Requires OAuth 2.0 setup.
    """
    if not GOOGLE_LIBS_AVAILABLE:
        print("POST_SCHEDULER_INFO: Google client libraries not available. Cannot get Blogger service.")
 main
        return None

    blogger_settings = settings.get('posting_platforms', {}).get('blogger', {})
    if not blogger_settings.get('enabled', False):
        console.print("[blue]POST_SCHEDULER_INFO:[/blue] Blogger posting is not enabled in settings.yaml.")
        return None

    console.print("[blue]POST_SCHEDULER_INFO:[/blue] Attempting to get Blogger service...")
=======
        print("POST_SCHEDULER_INFO: Blogger posting is not enabled in settings.yaml.")
        return None

    print("POST_SCHEDULER_INFO: Attempting to get Blogger service...")
 main

    token_path = blogger_settings.get('oauth_token_file', 'blogger_token.json')
    client_secrets_path = blogger_settings.get('client_secrets_file', 'client_secret_blogger.json')

    creds = None
    if os.path.exists(token_path):
        try:
            creds = UserCredentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/blogger'])
        except Exception as e:
            console.print(f"[yellow]POST_SCHEDULER_WARNING:[/yellow] Could not load token from {token_path}: {e}. Need to re-authenticate.")
=======
            print(f"POST_SCHEDULER_WARNING: Could not load token from {token_path}: {e}. Need to re-authenticate.")
 main

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                console.print("[blue]POST_SCHEDULER_INFO:[/blue] Credentials expired, attempting to refresh...")
                creds.refresh(google.auth.transport.requests.Request())
                console.print("[green]POST_SCHEDULER_INFO:[/green] Credentials refreshed successfully.")
                with open(token_path, 'w') as token_file:
                    token_file.write(creds.to_json())
                console.print(f"[blue]POST_SCHEDULER_INFO:[/blue] Refreshed token saved to {token_path}")
            except Exception as e:
                console.print(f"[bold red]POST_SCHEDULER_ERROR:[/bold red] Could not refresh token: {e}. Manual re-authentication needed.")
                creds = None
        else:
            console.print(f"[blue]POST_SCHEDULER_INFO:[/blue] No valid token found at {token_path} or token cannot be refreshed.")
            if not os.path.exists(client_secrets_path):
                console.print(f"[bold red]POST_SCHEDULER_ERROR:[/bold red] Client secrets file ('{client_secrets_path}') not found. Cannot initiate OAuth flow.")
                console.print("[blue]Please download your client_secret.json from Google Cloud Console for your OAuth 2.0 client ID and place it at the configured path.[/blue]")
                return None

            console.print(f"[blue]POST_SCHEDULER_INFO:[/blue] Attempting to initiate OAuth flow using {client_secrets_path}. This may require user interaction if run in a non-interactive environment for the first time.")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, ['https://www.googleapis.com/auth/blogger'])
                console.print("[yellow]POST_SCHEDULER_INFO: Please follow the instructions in your browser to authorize access.[/yellow]")
                console.print("[blue]POST_SCHEDULER_INFO: Waiting for OAuth authorization...[/blue]")
                creds = flow.run_local_server(port=0)
                with open(token_path, 'w') as token_file:
                     token_file.write(creds.to_json())
                console.print(f"[green]POST_SCHEDULER_INFO:[/green] New token obtained and saved to {token_path}")
            except Exception as e:
                console.print(f"[bold red]POST_SCHEDULER_ERROR:[/bold red] Failed to complete OAuth flow: {e}")
                console.print("[blue]Ensure you have a valid client_secrets.json and that the environment allows browser interaction for the first auth.[/blue]")
                return None

    if not creds:
        console.print("[bold red]POST_SCHEDULER_ERROR:[/bold red] Blogger authentication failed. Cannot get service.")
=======
                print("POST_SCHEDULER_INFO: Credentials expired, attempting to refresh...")
                creds.refresh(google.auth.transport.requests.Request())
                print("POST_SCHEDULER_INFO: Credentials refreshed successfully.")
                # Save the refreshed token
                with open(token_path, 'w') as token_file:
                    token_file.write(creds.to_json())
                print(f"POST_SCHEDULER_INFO: Refreshed token saved to {token_path}")
            except Exception as e:
                print(f"POST_SCHEDULER_ERROR: Could not refresh token: {e}. Manual re-authentication needed.")
                creds = None
        else:
            print(f"POST_SCHEDULER_INFO: No valid token found at {token_path} or token cannot be refreshed.")
            if not os.path.exists(client_secrets_path):
                print(f"POST_SCHEDULER_ERROR: Client secrets file ('{client_secrets_path}') not found. Cannot initiate OAuth flow.")
                print("Please download your client_secret.json from Google Cloud Console for your OAuth 2.0 client ID and place it at the configured path.")
                return None

            print(f"POST_SCHEDULER_INFO: Attempting to initiate OAuth flow using {client_secrets_path}. This may require user interaction if run in a non-interactive environment for the first time.")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, ['https://www.googleapis.com/auth/blogger'])
                print("POST_SCHEDULER_INFO: Please follow the instructions in your browser to authorize access.")
                print("POST_SCHEDULER_INFO: Waiting for OAuth authorization...")
                creds = flow.run_local_server(port=0)
                with open(token_path, 'w') as token_file:
                     token_file.write(creds.to_json())
                print(f"POST_SCHEDULER_INFO: New token obtained and saved to {token_path}")
            except Exception as e:
                print(f"POST_SCHEDULER_ERROR: Failed to complete OAuth flow: {e}")
                print("Ensure you have a valid client_secrets.json and that the environment allows browser interaction for the first auth.")
                return None

    if not creds: # Final check
        print("POST_SCHEDULER_ERROR: Blogger authentication failed. Cannot get service.")
 main
        return None

    try:
        service = build('blogger', 'v3', credentials=creds, static_discovery=False)
        console.print("[green]POST_SCHEDULER_INFO:[/green] Blogger service client created successfully.")
        return service
    except Exception as e:
        console.print(f"[bold red]POST_SCHEDULER_ERROR:[/bold red] Failed to build Blogger service: {e}")
        return None

def post_to_blogger(service, settings, title, content_html, labels=None, affiliate_link_override=None, image_path_for_post=None):
    if not service:
        console.print("[bold red]POST_SCHEDULER_ERROR:[/bold red] Blogger service not available, cannot post.")
=======
        print("POST_SCHEDULER_INFO: Blogger service client created successfully.")
        return service
    except Exception as e:
        print(f"POST_SCHEDULER_ERROR: Failed to build Blogger service: {e}")
        return None

def post_to_blogger(service, settings, title, content_html, labels=None, affiliate_link_override=None, image_path_for_post=None):
    """
    Posts an article to a Blogger blog.
    """
    if not service:
        print("POST_SCHEDULER_ERROR: Blogger service not available, cannot post.")
 main
        return False

    blogger_settings = settings.get('posting_platforms', {}).get('blogger', {})
    blog_id = blogger_settings.get('blog_id')

    if not blog_id or blog_id == "YOUR_BLOGGER_BLOG_ID":
        console.print(f"[bold red]POST_SCHEDULER_ERROR:[/bold red] Blogger Blog ID not configured or is set to placeholder in settings.yaml (current: '{blog_id}'). Cannot post.")
        return False

    console.print(f"[blue]POST_SCHEDULER_INFO:[/blue] Preparing to post to Blogger blog ID: {blog_id}...")
    if affiliate_link_override:
        console.print(f"[blue]POST_SCHEDULER_INFO:[/blue] Using affiliate link override: {affiliate_link_override} (expected to be already embedded in content_html).")
    if image_path_for_post:
        console.print(f"[blue]POST_SCHEDULER_INFO:[/blue] Image path '{image_path_for_post}' was provided for this post. (Image embedding logic TBD).")
=======
    if not blog_id or blog_id == "YOUR_BLOGGER_BLOG_ID": # Check for placeholder
        print(f"POST_SCHEDULER_ERROR: Blogger Blog ID not configured or is set to placeholder in settings.yaml (current: '{blog_id}'). Cannot post.")
        return False

    print(f"POST_SCHEDULER_INFO: Preparing to post to Blogger blog ID: {blog_id}...")
    if affiliate_link_override:
        print(f"POST_SCHEDULER_INFO: Using affiliate link override: {affiliate_link_override} (already embedded in content_html).")
    if image_path_for_post:
        print(f"POST_SCHEDULER_INFO: Image path '{image_path_for_post}' was provided for this post. (Image embedding logic TBD).")
        # Future: Add logic here to upload image to Blogger and insert into content_html if possible,
        # or use a pre-uploaded image URL if that's the workflow.
        # For now, content_html is assumed to be final or image is handled externally.
 main

    body = {
        "kind": "blogger#post",
        "blog": {"id": blog_id},
        "title": title,
        "content": content_html,
=======
        "content": content_html, # Assumed to have the correct affiliate link already
 main
    }
    if labels:
        body["labels"] = labels

    try:
        posts_service = service.posts()
        request = posts_service.insert(blogId=blog_id, body=body, isDraft=False)
        post = request.execute()
        console.print(f"[green]POST_SCHEDULER_SUCCESS:[/green] Successfully posted to Blogger. Post ID: {post['id']}, URL: {post.get('url', 'N/A')}")
        return True
    except HttpError as error:
        console.print(f"[bold red]POST_SCHEDULER_ERROR:[/bold red] An HTTP error {error.resp.status} occurred while posting to Blogger: {error._get_reason()}")
        console.print(f"[bold red]Detailed error:[/bold red] {error.content}")
    except Exception as e:
        console.print(f"[bold red]POST_SCHEDULER_ERROR:[/bold red] An unexpected error occurred while posting to Blogger: {e}")
=======
        print(f"POST_SCHEDULER_SUCCESS: Successfully posted to Blogger. Post ID: {post['id']}, URL: {post.get('url', 'N/A')}")
        return True
    except HttpError as error:
        print(f"POST_SCHEDULER_ERROR: An HTTP error {error.resp.status} occurred while posting to Blogger: {error._get_reason()}")
        print(f"Detailed error: {error.content}")
    except Exception as e:
        print(f"POST_SCHEDULER_ERROR: An unexpected error occurred while posting to Blogger: {e}")
 main
    return False

# --- WordPress Integration (Placeholder) ---
def post_to_wordpress(settings, title, content_html, affiliate_link_override=None, image_path_for_post=None):
    wp_settings = settings.get('posting_platforms', {}).get('wordpress', {})
    if not wp_settings.get('enabled', False):
        console.print("[blue]POST_SCHEDULER_INFO:[/blue] WordPress posting is not enabled in settings.yaml.")
        return False

    site_url = wp_settings.get('site_url', 'https://your-wordpress-site.com') # Provide default to avoid error if key missing
    config_username = wp_settings.get('username', 'your_wordpress_username')
    password_env_var = wp_settings.get('password_env_var')
    config_password = os.environ.get(password_env_var) if password_env_var else None

    current_username = config_username
    current_password = config_password

    # Check if settings are placeholders or password is not set
    if site_url == "https://your-wordpress-site.com" or \
       current_username == "your_wordpress_username" or \
       not current_password:
        console.print(f"[yellow]WARN:[/yellow] WordPress settings incomplete or password for user '{current_username}' not found via env var '{password_env_var}'.")
        dynamic_creds = prompt_for_credentials("WordPress")
        if dynamic_creds:
            current_username = dynamic_creds['username']
            current_password = dynamic_creds['password']
            console.print(f"[blue]INFO:[/blue] Using dynamically provided credentials for WordPress user '{current_username}'.")
        else:
            console.print(f"[bold red]ERROR:[/bold red] Cannot proceed with WordPress posting for '{current_username}' without credentials.")
            return False

    console.print(f"[blue]POST_SCHEDULER_INFO (Placeholder):[/blue] Would attempt to post to WordPress site: [link={site_url}]{site_url}[/link] as user [b]{current_username}[/b]")
    console.print(f"[blue]POST_SCHEDULER_INFO (Placeholder):[/blue] Title: {title[:50]}...")
    if affiliate_link_override:
        console.print(f"[blue]POST_SCHEDULER_INFO (Placeholder):[/blue] WordPress post would use affiliate link: {affiliate_link_override} (expected to be in content_html)")
    if image_path_for_post:
        console.print(f"[blue]POST_SCHEDULER_INFO (Placeholder):[/blue] WordPress post would attempt to use image: {image_path_for_post}")
    console.print("[cyan]POST_SCHEDULER_INFO (Placeholder):[/cyan] WordPress posting logic not yet implemented.")
    return False

# --- Social Media Posting (Placeholder) ---
def post_to_social_media(settings, text_content, image_path=None, affiliate_link_override=None):
    console.print(Panel("(Placeholder) Social Media Posting", title="[dim]Social Media[/dim]", expand=False))
    console.print(f"[blue]Content:[/blue] {text_content[:100]}...")
    if image_path:
        console.print(f"[blue]With image:[/blue] {image_path}")
    if affiliate_link_override:
        console.print(f"[blue]With link:[/blue] {affiliate_link_override}")
    console.print("[cyan]Social media posting logic not yet implemented.[/cyan]")
    return False

if __name__ == "__main__":
    console.print(Panel("Post Scheduler Script", title="[bold magenta]Agent Automation[/bold magenta]", subtitle="[dim]Initiating Workflow[/dim]"))
    current_settings = load_settings()

    if not current_settings:
        console.print("[bold red]FATAL:[/bold red] Exiting due to configuration loading failure.")
    else:
        console.print("[green]INFO:[/green] Configuration loaded successfully.")

        simulated_affiliate_link = "https://test.bybit.com/qr-ref"
        simulated_image_path = "ByBit.png"
        console.print(f"[blue]MAIN_TEST_INFO:[/blue] Using simulated affiliate link: [link={simulated_affiliate_link}]{simulated_affiliate_link}[/link]")
        console.print(f"[blue]MAIN_TEST_INFO:[/blue] Using simulated image path: [b]{simulated_image_path}[/b]")

        blogger_config = current_settings.get('posting_platforms', {}).get('blogger', {})
        if blogger_config.get('enabled', False) and GOOGLE_LIBS_AVAILABLE:
            console.print(Panel("Testing Blogger Posting", title="[cyan]Blogger Test[/cyan]", expand=False))

            client_secrets_file = blogger_config.get('client_secrets_file', 'client_secret_blogger.json')
            if not os.path.exists(client_secrets_file):
                console.print(f"[yellow]POST_SCHEDULER_WARNING:[/yellow] '{client_secrets_file}' not found. Creating a dummy one for structural testing ONLY.")
                console.print("[yellow]Real Blogger posting WILL FAIL without a valid client secrets file from Google Cloud Console.[/yellow]")
=======
    """Placeholder for WordPress posting functionality."""
    wp_settings = settings.get('posting_platforms', {}).get('wordpress', {})
    if not wp_settings.get('enabled', False):
        print("POST_SCHEDULER_INFO: WordPress posting is not enabled in settings.yaml.")
        return False

    site_url = wp_settings.get('site_url')
    username = wp_settings.get('username')
    password_env_var = wp_settings.get('password_env_var')

    if site_url == "https://your-wordpress-site.com" or username == "your_wordpress_username" or not password_env_var:
        print(f"POST_SCHEDULER_ERROR: WordPress settings (site_url, username, password_env_var) are placeholders or missing. Cannot post.")
        return False

    password = os.environ.get(password_env_var)
    if not password:
        print(f"POST_SCHEDULER_ERROR: WordPress password environment variable '{password_env_var}' not set. Cannot post.")
        return False

    print(f"POST_SCHEDULER_INFO: (Placeholder) Would attempt to post to WordPress site: {site_url}")
    print(f"POST_SCHEDULER_INFO: Title: {title[:50]}...")
    if affiliate_link_override:
        print(f"POST_SCHEDULER_INFO: (Placeholder) WordPress post would use affiliate link: {affiliate_link_override} (expected to be in content_html)")
    if image_path_for_post:
        print(f"POST_SCHEDULER_INFO: (Placeholder) WordPress post would attempt to use image: {image_path_for_post}")
    print("POST_SCHEDULER_INFO: WordPress posting logic not yet implemented.")
    return False

# --- Social Media Posting (Placeholder) ---
def post_to_social_media(settings, text_content, image_path=None, affiliate_link_override=None): # Renamed link to affiliate_link_override
    """Placeholder for social media posting functionality."""
    print("POST_SCHEDULER_INFO: (Placeholder) Would attempt to post to social media platforms.")
    print(f"POST_SCHEDULER_INFO: Content: {text_content[:100]}...")
    if image_path:
        print(f"POST_SCHEDULER_INFO: With image: {image_path}")
    if affiliate_link_override: # Updated parameter name
        print(f"POST_SCHEDULER_INFO: With link: {affiliate_link_override}")
    print("POST_SCHEDULER_INFO: Social media posting logic not yet implemented.")
    return False

if __name__ == "__main__":
    print("--- Post Scheduler Script ---")
    current_settings = load_settings()

    if not current_settings:
        print("POST_SCHEDULER_INFO: Exiting due to configuration loading failure.")
    else:
        print("POST_SCHEDULER_INFO: Configuration loaded successfully.")

        # Simulated dynamic values for testing
        simulated_affiliate_link = "https://test.bybit.com/qr-ref"
        simulated_image_path = "ByBit.png" # Assuming this image exists at the repo root for testing
        print(f"MAIN_TEST_INFO: Using simulated affiliate link: {simulated_affiliate_link}")
        print(f"MAIN_TEST_INFO: Using simulated image path: {simulated_image_path}")


        blogger_config = current_settings.get('posting_platforms', {}).get('blogger', {})
        if blogger_config.get('enabled', False) and GOOGLE_LIBS_AVAILABLE:
            print("\n--- Testing Blogger Posting ---")

            client_secrets_file = blogger_config.get('client_secrets_file', 'client_secret_blogger.json')
            if not os.path.exists(client_secrets_file):
                print(f"POST_SCHEDULER_WARNING: '{client_secrets_file}' not found. Creating a dummy one for structural testing ONLY.")
                print("POST_SCHEDULER_WARNING: Real Blogger posting WILL FAIL without a valid client secrets file from Google Cloud Console.")
 main
                try:
                    with open(client_secrets_file, 'w') as cs_file:
                        cs_file.write('{"installed":{"client_id":"YOUR_CLIENT_ID.apps.googleusercontent.com","project_id":"YOUR_PROJECT_ID","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_secret":"YOUR_CLIENT_SECRET","redirect_uris":["http://localhost"]}}')
                except IOError as e:
                    console.print(f"[bold red]POST_SCHEDULER_ERROR:[/bold red] Could not write dummy client secrets file: {e}")
=======
                    print(f"POST_SCHEDULER_ERROR: Could not write dummy client secrets file: {e}")
 main

            blogger_service_client = get_blogger_service(current_settings)
            if blogger_service_client:
                example_title = f"Test Post via Agent @ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
=======
                # Ensure the example_content_html uses the simulated_affiliate_link for the test
 main
                example_content_html = (
                    "<p>This is a <b>test post</b> generated automatically by the AI Marketing Agent.</p>"
                    "<p>It includes an affiliate link: <a href='{link}'>{link_text}</a></p>"
                    "<p><em>{disclaimer}</em></p>"
                ).format(
                    link=simulated_affiliate_link,
=======
                    link=simulated_affiliate_link, # Use the simulated link for test content
 main
                    link_text="Check out Bybit via QR!",
                    disclaimer=current_settings.get('compliance',{}).get('risk_disclaimer','Invest responsibly.')
                )
                example_labels = ["Test", "AI Agent", current_settings.get('target_keywords',["crypto"])[0]]

                console.print("[blue]POST_SCHEDULER_INFO:[/blue] Attempting test post to Blogger.")
                console.print("[yellow]NOTE: This will likely require manual OAuth browser interaction if 'blogger_token.json' is not present or invalid.[/yellow]")
=======
                print("POST_SCHEDULER_INFO: Attempting test post to Blogger.")
                print("POST_SCHEDULER_INFO: NOTE: This will likely require manual OAuth browser interaction if 'blogger_token.json' is not present or invalid.")
 main

                post_to_blogger(
                    blogger_service_client,
                    current_settings,
                    example_title,
                    example_content_html,
                    labels=example_labels,
                    affiliate_link_override=simulated_affiliate_link,
                    image_path_for_post=simulated_image_path
                )
            else:
                console.print("[blue]INFO:[/blue] Blogger service could not be initialized. Test post skipped.")
        else:
            console.print(Panel("Blogger posting is disabled or Google libraries are missing. Skipping Blogger test.", title="[dim]Blogger Test Skipped[/dim]", expand=False))

        wordpress_config = current_settings.get('posting_platforms', {}).get('wordpress', {})
        if wordpress_config.get('enabled', False):
            console.print(Panel("Testing WordPress Posting (Placeholder)", title="[cyan]WordPress Test[/cyan]", expand=False))
            post_to_wordpress(
                current_settings,
                "WP Test Post",
                f"<p>Test content for WordPress with link: {simulated_affiliate_link}</p>", # Formatted string
=======
                    affiliate_link_override=simulated_affiliate_link, # Pass the override
                    image_path_for_post=simulated_image_path      # Pass the image path
                )
            else:
                print("POST_SCHEDULER_INFO: Blogger service could not be initialized. Test post skipped.")
        else:
            print("\nPOST_SCHEDULER_INFO: Blogger posting is disabled or Google libraries are missing. Skipping Blogger test.")

        wordpress_config = current_settings.get('posting_platforms', {}).get('wordpress', {})
        if wordpress_config.get('enabled', False):
            print("\n--- Testing WordPress Posting (Placeholder) ---")
            post_to_wordpress(
                current_settings,
                "WP Test Post",
                "<p>Test content for WordPress with link: {simulated_affiliate_link}</p>",
 main
                affiliate_link_override=simulated_affiliate_link,
                image_path_for_post=simulated_image_path
            )
        else:
            console.print(Panel("WordPress posting is disabled. Skipping WordPress test.", title="[dim]WordPress Test Skipped[/dim]", expand=False))

        console.print(Panel("Testing Social Media Posting (Placeholder)", title="[cyan]Social Media Test[/cyan]", expand=False))
        post_to_social_media(
            current_settings,
            "Check out this cool update! #Bybit #Crypto",
            image_path=simulated_image_path,
            affiliate_link_override=simulated_affiliate_link
        )

    console.print(Panel("Post Scheduler Script Finished", style="bold green", padding=(1,2)))
=======
            print("\nPOST_SCHEDULER_INFO: WordPress posting is disabled. Skipping WordPress test.")

        print("\n--- Testing Social Media Posting (Placeholder) ---")
        post_to_social_media(
            current_settings,
            "Check out this cool update! #Bybit #Crypto",
            image_path=simulated_image_path, # Existing parameter name is fine
            affiliate_link_override=simulated_affiliate_link # Pass the override
        )

        print("\n--- Post Scheduler Script Finished ---")
 main
