from django import forms

class ReviewForm(forms.Form):
    review_text = forms.CharField(
        label='Your review',
        widget=forms.Textarea(attrs={'rows': 5, 'cols': 40}),
        max_length=2000,
        help_text='Enter your review of the film.'
    )