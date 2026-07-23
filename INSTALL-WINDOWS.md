# pwvault for Windows  (setup guide)

Read section 5 before you start using it seriously,  it is the one part that
can cost you your passwords if you skip it.

---

## 1. What this program is

pwvault keeps your website logins in one encrypted file on your own computer.
Nothing is uploaded anywhere, and there is no account to sign up for.

To open your passwords you need **two things together**:

|                             |                                                                         |
| --------------------------- | ----------------------------------------------------------------------- |
| **A master password** | One password you memorise. Never written down anywhere on the computer. |
| **A key file**        | A small file called`pwvault.key` that the program makes for you.      |

Someone who steals your computer and finds the vault file still cannot read
your passwords without both. That is the whole point of the design.

---

## 2. Download it

You will be sent a file called **`pwvault.exe`**. Save it somewhere you can
find it again , your **Desktop** or your **Documents** folder is fine.

That single file *is* the program. There is no installer and nothing gets added
to your system.

> **Tip:** right-click `pwvault.exe` → **Pin to Start** (or **Show more options →
> Send to → Desktop**) so it's easy to reach later.

---

## 3. The blue warning box on first run

The first time you double-click `pwvault.exe`, Windows will very likely show a
blue box saying:

> **Windows protected your PC**
> Microsoft Defender SmartScreen prevented an unrecognised app from starting.

**This is normal and expected.** It does not mean anything is wrong with the
program. Windows shows this for every app that hasn't been through Microsoft's
paid code-signing process, which costs a few hundred euros a year.

To continue:

1. Click **More info** (the small link in the box).
2. Click the **Run anyway** button that appears.

You should only have to do this once.

> Only do this for the file your friend sent you directly. This advice is not
> a general rule for every warning Windows shows you.

---

## 4. Setting up your vault (first run only)

When the program opens, it will walk you through three steps.

**Step 1 — Choose your master password.**
Type it twice. Make it something you will genuinely remember, because there is
no reset and no recovery. A short sentence works well:
`my dog ate the blue sofa` is both easy to remember and very hard to guess.
Minimum 8 characters.

**Step 2 — Save the key file.**
A "Save as" window appears asking where to put `pwvault.key`. Just press
**Save** to accept the suggested location for now — you will make a proper
backup in the next step.

**Step 3 — Read the message that appears.** It tells you exactly where your key
file was saved. Note that location down.

That's it. Your vault is ready.

---

## 5. ⚠️ Back up your key file — do this now

**If you lose the key file, every password in your vault is gone permanently.**
Not "hard to recover" — gone. Nobody, including whoever gave you this program,
can get them back :) The same is true if you forget your master password.

So before you put anything important in:

1. Plug in a **USB stick**.
2. In pwvault, click the **File** menu → **Back up key file…**
3. Choose the USB stick and click **Save**.
4. Put the USB stick somewhere safe that isn't next to your computer.

Do the same with **File → Back up vault…** every few weeks. The vault backup
stays encrypted, so it is safe to keep on a USB stick or in cloud storage —
it is useless to anyone without your master password and key file.

Two separate risks, two separate protections:

- Someone **steals** your files → they cannot read them (that's the encryption).
- Your **hard drive dies** → you need the backups (that's this section).

---

## 6. Using it day to day

**Opening it.** Double-click `pwvault.exe`, type your master password, click
**Unlock**. It remembers where your key file is, so you normally don't need to
touch that box.

> If your key file is on a USB stick, plug the stick in *before* clicking
> Unlock, then use **Browse…** to point at it.

**Saving a new login.** Click **Add entry**. Fill in the name (e.g. `Amazon`),
your username or email, and the password. Click **Generate** if you want the
program to invent a strong random password for you — this is the single best
habit to get into, because it means a leak at one website cannot expose any of
your other accounts. The website and notes fields are optional.

> **Generate** lets you set how long the password should be and whether to
> include digits and symbols. Some websites refuse symbols; switch them off
> there. **Try another** rolls a new one until you are happy with it.

**Getting a password back.** Click the entry in the list, then **Copy
password**. Switch to your browser and paste with **Ctrl+V**. The clipboard
wipes itself after 20 seconds so the password isn't left lying around.

**Finding something.** Type in the **Search** box at the top. It looks through
names, usernames, websites, and notes as you type.

**Changing or removing.** Select an entry, then **Edit** or **Delete**.
Double-clicking an entry also opens it for editing.

**Locking.** Click **Lock** when you step away. It also locks itself
automatically after 5 minutes of no activity, and whenever you close the
window.

**Dark mode.** The **Mode** menu at the top switches between light and dark;
it remembers your choice for next time.

**Stuck?** The **Help** menu answers the common questions — what the key file
is for, how to back up, what happens if you forget your master password.

---

## 7. Where your files are

Use **Help → Where are my files?** inside the program to see the exact
locations on your machine. By default:

|                                  |                                                      |
| -------------------------------- | ---------------------------------------------------- |
| Vault (your encrypted passwords) | `C:\Users\<you>\AppData\Roaming\pwvault\vault.enc` |
| Key file                         | Wherever you chose to save it in step 2              |

`AppData` is a hidden folder. You do not need to go there — the program handles
it, and the menu items in section 5 are the easy way to make copies.

---

## 8. If something goes wrong

**"Wrong password or key file"**
Either the master password was typed wrong (check Caps Lock), or the key file
selected isn't the right one. If you have several copies of `pwvault.key`, make
sure you are using one made for *this* vault.

**"Key file not found"**
The file has been moved, renamed, or is on a USB stick that isn't plugged in.
Plug it in, click **Browse…**, and select the file.

**The window doesn't open at all**
Check that Windows didn't quarantine the file: right-click `pwvault.exe` →
**Properties**. If there is an **Unblock** checkbox near the bottom, tick it and
click **OK**.

**You forgot your master password**
There is genuinely nothing that can be done. The vault is encrypted with a key
derived from that password; without it the data is unreadable noise. You would
have to start a new vault and reset your passwords at each website.

---

## 9. Moving to a new computer

1. On the old PC: **File → Back up vault…** and **File → Back up key file…**,
   both onto a USB stick.
2. On the new PC: copy `pwvault.exe` across and run it.
3. It will offer to create a new vault — **don't**. Close it, and instead copy
   your backed-up `vault.enc` into
   `C:\Users\<you>\AppData\Roaming\pwvault\` (create the `pwvault` folder if it
   isn't there). Paste the folder path into the address bar of File Explorer to
   get there quickly.
4. Run pwvault again, point it at your key file with **Browse…**, and enter
   your master password.
