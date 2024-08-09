export class AuthService {
    login(username: string, password: string) {
        // implement your login logic here
        // for example, you can make an API call to your backend
        return fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
    }
}

export default new AuthService();
