import urllib.request, urllib.error, urllib.parse
from bs4 import BeautifulSoup

# Concordia uses CloudFlare, which rejects GET requests that are lacking a
# User-Agent string. We need to create a string that mimics a web browser but
# then to be nice we will add our custom user agent string. And yes, typical
# user agents really do reference all these various versions of web browsers
mozilla = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
applewebkit = 'AppleWebKit/537.36 (KHTML, like Gecko)'
chrome = 'Chrome/97.0.4692.71 Safari/537.36'
safari = 'Safari/537.36'
politeness = '** Class exercise for CSC432 (see Prof Tallman) **'
custom_user_agent = f"{mozilla} {applewebkit} {chrome} {safari} {politeness}"

# Slightly different format for our URL request
url = 'https://www.cui.edu/hr/employee-directory'
headers = {
    'User-Agent': custom_user_agent
}
request = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(request) as response:
    content = response.read().decode('UTF-8')

# Our web pages are nested to an annoying degree, but this is common for many 
# website designer packages
soup = BeautifulSoup(content, "html.parser")
html = soup.html
body = html.body
form = body.contents[1]
main = form.contents[38]
data = main.contents[3]
div = data.contents[1]
div = div.contents[1]
div = div.contents[0]
div = div.contents[1]
div = div.contents[1]
div = div.contents[1]
div = div.contents[2]
div = div.contents[1]
div = div.contents[1]
div = div.contents[1]
tbody = div.contents[3]


def extract_row(tr):
    ''' Extracts a Concordia staff member's name, phone number, and email from the directory '''
    # Raw HTML:
    #   <tr>
    #   <td style="text-align: center;">Redira, Alicia</td>
    #   <td style="text-align: center;">949-214-3842</td>
    #   <td style="text-align: center;"><a href="/cdn-cgi/l/email-protection#bbdad7d2d8d2da95c9dedfd2c9dafbd8ced295dedfce" target="blank"><span class="__cf_email__" data-cfemail="a9c8c5c0cac0c887dbcccdc0dbc8e9cadcc087cccddc">[email&#160;protected]</span></a></td>
    #   </tr>
    # tr.contents
    #   0: '\n' (e.g., the end of the <tr>)
    #   1: '<td ...>Redira, Alicia</td>'
    #   2: '\n' (e.g., the end of the <tr>)
    #   3: '<td ...>949-214-3842</td>'
    #   4: '\n' (e.g., the end of the <tr>)
    #   5: '<td ...><a href=...><span ...>[email protected]</span></a></td>'
    #   6: '\n' (e.g., the end of the <tr>)
    
    if len(tr) > 5:
        name = tr.contents[1].get_text(strip=True)
        phone = tr.contents[3].get_text(strip=True)
        span = tr.contents[5].span
        # CloudFlare email protection encodes email addresses by XORing a random byte with the email address
        # The "random byte" is placed at the beginning of the string so we just need to pull off the first
        # byte and then use it to XOR every byte that follows. Wallah: the email addresses.
        # 
        # See https://developers.cloudflare.com/waf/tools/scrape-shield/email-address-obfuscation/
        # and https://usamaejaz.com/cloudflare-email-decoding/
        #
        # How did we find this? The decoding script is part of the webpage. Inspect the webpage and search
        # for 'data-cfemail' one of the references is to 'email-decode.min.js' If you inpsect this function
        # you'll find code similar to what I have below.
        if span and span.has_attr('data-cfemail'):
            email = ''
            cf_email = span['data-cfemail']
            key = int(cf_email[:2], 16)
            for i in range(2, len(cf_email), 2):
                cf_byte = cf_email[i:i+2]
                cf_byte = int(cf_byte, 16)
                email += chr(cf_byte ^ key)
        else:
            email = tr.contents[5].get_text(strip=True)
    else:
        name = phone = email = 'unknown'

    return name, phone, email

def validate_phone(number: str):
    if len(number.strip()) == 0:
        return ''

    number = number.split('-')

    if len(number) != 3:
        return ''
    
    return f'({number[0].strip()}) {number[1].strip()}-{number[2].strip()}'


tbodys = soup.find_all('tbody')

# there are two tbody tags - one for the first person, and one for the rest

# deal with first body tag
name, phone, _ = extract_row(tbodys[0].tr)
print(f"{name:<10}{validate_phone(phone):>30}")

# deal with the second body tag with the rest of the people
for tr in tbodys[1].find_all('tr'):
    name, phone, _ = extract_row(tr)
    print(f"{name:<10}{validate_phone(phone):>30}")
